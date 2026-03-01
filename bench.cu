#include "deepeeRT/deepeeRT.h"
#include "miniScene/Scene.h"
#include <fstream>

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

// __global__ void g_clearHits(DPRHit *hits, int N)
// {
//   int tid = threadIdx.x+blockIdx.x*blockDim.x;
//   if (tid >= N) return;
//   hits[tid].primID = -1;
// }

// void clearHits(DPRHit *d_hits, int N)
// {
//   int bs = 128;
//   int nb = divRoundUp(N,bs);
//   g_clearHits<<<nb,bs>>>(d_hits,N);
// }


std::vector<std::vector<DPRRay>> loadRays(const std::string &fileName)
{
  std::cout << "loading rays from " << fileName << std::endl;
  std::ifstream in(fileName.c_str(),std::ios::binary);
  if (!in.good())
    throw std::runtime_error("could not open rays file!?");
  std::vector<std::vector<DPRRay>> ret;
  while (in.good() && !in.eof()) {
    size_t numRays = 0;
    in.read((char *)&numRays,sizeof(numRays));
    assert(numRays != 0);
    ret.push_back({});
    ret.back().resize(numRays);
    in.read((char *)ret.back().data(),numRays*sizeof(DPRRay));
  }
  return ret;
}

DPRWorld specifyToDP(DPRContext ctx,
                       mini::Scene::SP miniModel)
{
  assert(ctx);

  std::map<mini::Object::SP,DPRGroup> dprGroupFor;

  std::vector<DPRGroup> instanceGroups;
  std::vector<mini::common::affine3d> instanceTransforms;
  uint64_t uniqueMeshIDs = 0;
  for (auto inst : miniModel->instances) {
    if (!dprGroupFor[inst->object]) {
      std::vector<DPRTriangles> meshes;
      for (auto miniMesh : inst->object->meshes) {
        DPRTriangles dt
          = dprCreateTrianglesDP(ctx,
                                 uniqueMeshIDs++,
                                 (DPRvec3*)miniMesh->vertices.data(),
                                 miniMesh->vertices.size(),
                                 (DPRint3*)miniMesh->indices.data(),
                                 miniMesh->indices.size());
        assert(dt);
        meshes.push_back(dt);
      }
      DPRGroup group
        = dprCreateTrianglesGroup(ctx,
                                  meshes.data(),
                                  meshes.size());
      assert(group);
      dprGroupFor[inst->object] = group;
    }
    instanceGroups.push_back(dprGroupFor[inst->object]);
    instanceTransforms.push_back(inst->xfm);
  }
  DPRWorld dprModel
    = dprCreateWorldDP(ctx,
                       instanceGroups.data(),
                       (DPRAffine*)instanceTransforms.data(),
                       instanceGroups.size());
  return dprModel;
}

void trace(DPRWorld dpModel,
           std::vector<DPRRay *> &devRayFronts,
           std::vector<DPRHit *> &devHitFronts,
           std::vector<int>      &devRayCounts)
{
  for (int i=0;i<devRayFronts.size();i++)
    dprTrace(dpModel,devRayFronts[i],devHitFronts[i],devRayCounts[i]);
}

int main(int ac, char **av)
{
  std::string inModelName = "";
  std::string inRaysName = "";
  std::string inHitsName = "";
  std::string outHitsName = "";
  double numSecsForMeasure = 10.;

  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-oh")
      outHitsName = av[++i];
    else if (arg == "-im")
      inModelName = av[++i];
    else
      throw std::runtime_error("unknown cmdline arg '"+arg+"'");
  }

  mini::Scene::SP miniModel = mini::Scene::load(inModelName);
  DPRContext ctx = dprContextCreate(DPR_CONTEXT_GPU,0);
  DPRWorld dpModel = specifyToDP(ctx,miniModel);

  std::vector<std::vector<DPRRay>> hostRayFronts = loadRays(inRaysName);
  std::vector<DPRRay *> devRayFronts;
  std::vector<DPRHit *> devHitFronts;
  std::vector<int>      devRayCounts;
  size_t numRaysInTotal = 0;
  for (auto &wave : hostRayFronts) {
    devRayFronts.push_back(upload<DPRRay>(wave));
    devHitFronts.push_back(alloc<DPRHit>(wave.size()));
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
  return 0;
}

