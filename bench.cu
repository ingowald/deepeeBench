#include "dprt/dprt.h"
#include "miniScene/Scene.h"
#include <fstream>
#include <thread>

using mini::common::prettyNumber;
using mini::common::prettyDouble;
using mini::common::getCurrentTime;

template<typename T>
T *upload(const std::vector<T> &t)
{
  T *d_t = 0;
  cudaMalloc((void **)&d_t,t.size()*sizeof(t[0]));
  cudaMemcpy(d_t,t.data(),t.size()*sizeof(t[0]),cudaMemcpyDefault);
  return d_t;
}

template<typename T>
T *alloc(size_t N)
{
  T *d_t = 0;
  cudaMalloc((void **)&d_t,N*sizeof(T));
  return d_t;
}

// __global__ void g_clearHits(DPRTHit *hits, int N)
// {
//   int tid = threadIdx.x+blockIdx.x*blockDim.x;
//   if (tid >= N) return;
//   hits[tid].primID = -1;
// }

// void clearHits(DPRTHit *d_hits, int N)
// {
//   int bs = 128;
//   int nb = divRoundUp(N,bs);
//   g_clearHits<<<nb,bs>>>(d_hits,N);
// }

namespace watchDog {
 int timeToTrigger = 60;
 bool shutdown = false;
 bool running = false;
 std::thread thread;
  
 void start() {
    running = true;
    shutdown = false;
    thread = std::thread([]() {
      int numSecsWaited = 0;
      while (true) {
        if (shutdown) return;
        sleep(1);
        if (shutdown) return;
        if (++numSecsWaited >= timeToTrigger) {
          std::cout << "WATCHDOG TRIGGERED!" << std::endl;
          _exit(0);
        }
      }
    });
  }
 void end()
  {
    shutdown = true;
    thread.join();
    running = false;
  }
}

  void usage(const std::string &error)
  {
    if (error != "") std::cout << "Error : " << error << "\n\n";
    std::cout << "./dpBench <flags>" << std::endl;
    std::cout << "/w flags:" << std::endl;
    std::cout << "  -irf inRaysFile.dprays" << std::endl;
    std::cout << "  -imf inModelsFile.dpmini" << std::endl;
    std::cout << "  --watchDog watchDogTimeInSeconds" << std::endl;
    exit(0);
  }


std::vector<std::vector<DPRTRay>> loadRays(const std::string &fileName)
{
  std::cout << "loading rays from " << fileName << std::endl;
  std::ifstream in(fileName.c_str(),std::ios::binary);
  if (!in.good())
    throw std::runtime_error("could not open rays file!?");
  std::vector<std::vector<DPRTRay>> ret;
  while (in.good() && !in.eof()) {
    PING;
    size_t numRays = 0;
    in.read((char *)&numRays,sizeof(numRays));
    if (!in.good()) break;
    PRINT(numRays);
    assert(numRays != 0);
    ret.push_back({});
    ret.back().resize(numRays);
    in.read((char *)ret.back().data(),numRays*sizeof(DPRTRay));
  }
  return ret;
}

DPRTModel toDPRT(DPRTContext ctx,
                       mini::Scene::SP miniModel)
{
  assert(ctx);

  std::map<mini::Object::SP,DPRTGroup> dprtGroupFor;

  std::vector<DPRTGroup> instanceGroups;
  std::vector<mini::common::affine3d> instanceTransforms;
  uint64_t uniqueMeshIDs = 0;
  for (auto inst : miniModel->instances) {
    if (!dprtGroupFor[inst->object]) {
      std::vector<DPRTTriangles> meshes;
      for (auto miniMesh : inst->object->meshes) {
        DPRTTriangles dt
          = dprtCreateTriangles(ctx,
                                 uniqueMeshIDs++,
                                 (DPRTvec3*)miniMesh->vertices.data(),
                                 miniMesh->vertices.size(),
                                 (DPRTint3*)miniMesh->indices.data(),
                                 miniMesh->indices.size());
        assert(dt);
        meshes.push_back(dt);
      }
      DPRTGroup group
        = dprtCreateTrianglesGroup(ctx,
                                  meshes.data(),
                                  meshes.size());
      assert(group);
      dprtGroupFor[inst->object] = group;
    }
    instanceGroups.push_back(dprtGroupFor[inst->object]);
    instanceTransforms.push_back(inst->xfm);
  }
  DPRTModel dprtModel
    = dprtCreateModel(ctx,
                       instanceGroups.data(),
                       (DPRTAffine*)instanceTransforms.data(),
                       instanceGroups.size());
  return dprtModel;
}



void trace(DPRTModel dpModel,
           std::vector<DPRTRay *> &devRayFronts,
           std::vector<DPRTHit *> &devHitFronts,
           std::vector<int>      &devRayCounts)
{
  watchDog::start();
  for (int i=0;i<devRayFronts.size();i++) 
    dprtTrace(dpModel,devRayFronts[i],devHitFronts[i],devRayCounts[i]);
  watchDog::end();
}

int main(int ac, char **av)
{
  std::string inModelName = "";
  std::string inRaysName = "";
  std::string inHitsName = "";
  std::string outHitsName = "";
  double numSecsForMeasure = 4.;

  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-oh")
      outHitsName = av[++i];
    else if (arg == "-imf")
      inModelName = av[++i];
    else if (arg == "-irf")
      inRaysName = av[++i];
    else if (arg == "--watchDog")
      watchDog::timeToTrigger = std::stoi(av[++i]);
    else
      throw std::runtime_error("unknown cmdline arg '"+arg+"'");
  }
  if (inModelName.empty())
    usage("no in model name (-imf) specified");
  if (inRaysName.empty())
    usage("no in rays-file name (-irf) specified");

  mini::Scene::SP miniModel = mini::Scene::load(inModelName);
  DPRTContext ctx = dprtContextCreate(DPRT_CONTEXT_GPU,0);
  DPRTModel dpModel = toDPRT(ctx,miniModel);

  std::vector<std::vector<DPRTRay>> hostRayFronts = loadRays(inRaysName);
  std::vector<DPRTRay *> devRayFronts;
  std::vector<DPRTHit *> devHitFronts;
  std::vector<int>       devRayCounts;
  size_t numRaysInTotal = 0;
  for (auto &wave : hostRayFronts) {
    devRayFronts.push_back(upload<DPRTRay>(wave));
    devHitFronts.push_back(alloc<DPRTHit>(wave.size()));
    numRaysInTotal += wave.size();
    devRayCounts.push_back(wave.size());
  }

  // first round for warmup
  trace(dpModel,devRayFronts,devHitFronts,devRayCounts);

  int numItsDone = 1;
  double timeForTheseIts = 0.;
  while (true) {
    std::cout << "running " << numItsDone << " iterations..." << std::endl;
    double t0 = getCurrentTime();
    for (int it=0;it<numItsDone;it++)
      trace(dpModel,devRayFronts,devHitFronts,devRayCounts);
    double t1 = getCurrentTime();
    timeForTheseIts = t1-t0;
    if (timeForTheseIts >= numSecsForMeasure)
      break;
    numItsDone *= 2;
  }

  size_t numAllRays = numRaysInTotal * numItsDone;
  std::cout << "traced " << numItsDone << " its of "
            << numRaysInTotal << " rays per it, in "
            << prettyDouble(timeForTheseIts) << "s, that's "
            << prettyDouble(numAllRays/timeForTheseIts) << "rays/s"
            << std::endl;
  std::cout << << "PERF_RPS " << prettyDouble(numAllRays/timeForTheseIts) << std::endl;
  return 0;
}

