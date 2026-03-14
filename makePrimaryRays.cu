// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dprt/dprt.h"
#include "miniScene/Scene.h"
#include <fstream>
#include <thread>

#if FORCE_BUG
# define dbg_x 965 // miss
#else
# define dbg_x 975 // hit
#endif
# define dbg_y 200



#define CUDA_SYNC_CHECK()                                       \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      throw std::runtime_error("fatal cuda error");             \
    }                                                           \
  }

extern int dprt_dbg_rayID;

namespace miniapp {
  std::thread wdThread;
  bool wdTerminate = false;
  using namespace mini;

  int watchDogTime = 0;
  double shift = 0.;
  
  struct {
    vec3d from = 0.;
    vec3d at = 0.;
    vec3d up { 0.,1.,0.};
    double fovy = 0.;
    double native_scale = 0.;
    bool do_ortho = false;
  } view;
  
  /*! this HAS to be the same data layout as DPRRay in deepee.h */
  struct Ray {
    vec3d origin;
    vec3d direction;
    double  tMin;
    double  tMax;
  };

  struct Camera {
    /*! generate ray for a given pixel/image plane coordinate. based on
      how the camera was created this could be either a orthogonal or
      a perspective camera (see Camera.cpp) */
    inline __device__
    Ray generateRay(vec2d pixel, double shift, bool dbg=false) const;

    struct {
      vec3d v,du,dv;
    } origin, direction;
  };

  Camera generateOrtho(vec2i imageRes, double shift)
  {
    Camera camera;
    vec3d center = view.from;
    vec3d dir = view.at - center;
    vec3d du = normalize(cross(dir,view.up));
    vec3d dv = normalize(cross(du,dir));
    camera.origin.dv = dv*view.native_scale/imageRes.y;
    camera.origin.du = du*view.native_scale/imageRes.y;
    camera.origin.v  = view.from
      - (shift * view.native_scale) * normalize(dir)
      - .5 *imageRes.x * camera.origin.du
      - .5 *imageRes.y * camera.origin.dv;
    camera.direction.v = dir;
    camera.direction.du = vec3d(0.);
    camera.direction.dv = vec3d(0.);
    return camera;
  }
  
  Camera generateCamera(vec2i imageRes,
                        // const box3d &bounds,
                        // const vec3d &from_dir,
                        // const vec3d &up,
                        double shift)
  {
    Camera camera;
    vec3d target = view.at;//bounds.center();
    vec3d from = view.from;//target + from_dir;
    vec3d direction = normalize(target-from);
    PRINT(target);
    PRINT(from);
    
    vec3d du = normalize(cross(direction,view.up));
    vec3d dv = normalize(cross(du,direction));
    
    double aspect = imageRes.x/double(imageRes.y);
    // double scale = length(bounds.size());

    // direction *= scale;
    // dv *= scale;
    // du *= scale*aspect;
    du *= aspect;

    // for testing: this is a perspective camera with all origins on a
    // point, and different ray directions each. this corresponds to
    // fovy=60
    PRINT(direction);
    PRINT(du);
    PRINT(dv);
    camera.direction.v = .5 * direction-.5*du-.5*dv;
    camera.direction.du = du * (1./imageRes.x);
    camera.direction.dv = dv * (1./imageRes.y);

    camera.origin.v = from;
    camera.origin.du = 0.;
    camera.origin.dv = 0.;
    camera.origin.v += (view.native_scale*shift) * vec3d(1.,.1,.01);

    PRINT(camera.origin.v);
    
    return camera;
  }

  void shadePretty(vec4f *m_pixels,
                   vec2i fbSize,
                   DPRTRay *h_rays,
                   DPRTHit *h_hits,
                   mini::Scene::SP scene,
                   std::vector<mini::Mesh::SP> &linearMeshes)
  {
    for (int i=0;i<fbSize.x*fbSize.y;i++) {
      DPRTHit hit = h_hits[i];
      vec3f pixel;
      if (hit.primID < 0) {
        pixel = vec3f(.8,.8,.9);
      } else {
        auto inst = scene->instances[hit.instID];
        auto mesh = linearMeshes[(int)hit.geomUserData];
        auto tri = mesh->indices[hit.primID];
        auto v0 = mesh->vertices[tri.x];
        auto v1 = mesh->vertices[tri.y];
        auto v2 = mesh->vertices[tri.z];
        vec3d N = normalize(cross(v1-v0,v2-v0));
        double c = std::abs(dot(N,normalize((vec3d&)h_rays[i].direction)));
        c = .2+c*.8;
        pixel = (float)c*(.5f+.5f*randomColor(hit.primID));
      }
      m_pixels[i].x = pixel.x;
      m_pixels[i].y = pixel.y;
      m_pixels[i].z = pixel.z;
      m_pixels[i].w = 1.f;
    }
  }
  

  /*! generate ray for a given pixel/image plane coordinate. based on
    how the camera was created this could be either a orthogonal or
    a perspective camera (see Camera.cpp) */
  inline __device__
  Ray Camera::generateRay(vec2d pixel, double shift, bool dbg) const
  {
    Ray ray;
    ray.origin
      = origin.v+pixel.x*origin.du+pixel.y*origin.dv;
    if (dbg)
      printf("camera org %f %f %f\n",
             (float)origin.v.x,
             (float)origin.v.y,
             (float)origin.v.z);

    ray.direction
      = normalize(direction.v+pixel.x*direction.du+pixel.y*direction.dv);
    ray.tMin = 0.;
    ray.tMax = INFINITY;
    return ray;
  }

  
  /*! helper function that creates a semi-random color from an ID */
  inline __host__ __device__ vec3f randomColor(int i)
  {
    const uint64_t FNV_offset_basis = 0xcbf29ce484222325ULL;
    const uint64_t FNV_prime = 0x10001a7;
    uint32_t v = (uint32_t)FNV_offset_basis;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;
    v = FNV_prime * v ^ i;

    int r = v >> 24;
    v = FNV_prime * v ^ i;
    int b = v >> 16;
    v = FNV_prime * v ^ i;
    int g = v >> 8;
    return vec3f((r&255)/255.f,
                 (g&255)/255.f,
                 (b&255)/255.f);
  }

  void getFrame(std::string up,
                vec3d &dx,
                vec3d &dy,
                vec3d &dz)
  {
    if (up == "z") {
      dx = {1.,0.,0.};
      dy = {0.,1.,0.};
      dz = {0.,0.,1.};
      return;
    }
    if (up == "y") {
      dx = {1.,0.,0.};
      dz = {0.,1.,0.};
      dy = {0.,0.,1.};
      return;
    }
    throw std::runtime_error("unhandled 'up'-specifier of '"+up+"'");
  }

  DPRTModel toDPRT(DPRTContext ctx,
                   mini::Scene::SP miniModel,
                   std::vector<mini::Mesh::SP> &linearMeshes,
                   double shift)
  {
    assert(ctx);

    std::map<mini::Object::SP,DPRTGroup> dprtGroupFor;

    std::vector<DPRTGroup> instanceGroups;
    std::vector<mini::common::affine3d> instanceTransforms;
    // uint64_t uniqueMeshIDs = 0;

    bool hasInstances = false;
    for (auto inst : miniModel->instances) {
      if (inst->xfm != affine3d()) {
        // PRINT(inst->xfm);
        hasInstances = true;
      }
    }
    
    for (auto inst : miniModel->instances) {
      if (!dprtGroupFor[inst->object]) {
        std::vector<DPRTTriangles> meshes;
        for (auto miniMesh : inst->object->meshes) {
          auto &vertices = miniMesh->vertices;
          if (!hasInstances)
            for (auto &v : vertices)
              v += (shift*view.native_scale) * vec3d(1.,.1,.01);
          uint64_t uniqueMeshID  = linearMeshes.size();
          DPRTTriangles dt
            = dprtCreateTriangles(ctx,
                                  uniqueMeshID,
                                  (DPRTvec3*)miniMesh->vertices.data(),
                                  miniMesh->vertices.size(),
                                  (DPRTint3*)miniMesh->indices.data(),
                                  miniMesh->indices.size());
          assert(dt);
          meshes.push_back(dt);
          linearMeshes.push_back(miniMesh);
        }
        DPRTGroup group
          = dprtCreateTrianglesGroup(ctx,
                                     meshes.data(),
                                     meshes.size());
        assert(group);
        dprtGroupFor[inst->object] = group;
      }
      instanceGroups.push_back(dprtGroupFor[inst->object]);
      auto xfm = inst->xfm;
      if (hasInstances)
        xfm.p += (shift*view.native_scale) * vec3d(1.,.1,.01);
      instanceTransforms.push_back(xfm);
    }
    // for (auto xfm : instanceTransforms)
    //   PRINT(xfm);
    DPRTModel dprtModel
      = dprtCreateModel(ctx,
                        instanceGroups.data(),
                        (DPRTAffine*)instanceTransforms.data(),
                        instanceGroups.size());
    return dprtModel;
  }

  
  __global__
  void g_shadeRays(vec4f *d_pixels,
                   DPRTRay *d_rays,
                   DPRTHit *d_hits,
                   vec2i fbSize)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;

    DPRTHit hit = d_hits[ix+iy*fbSize.x];
    vec3f color = randomColor(hit.primID >= 0
                              ? hit.primID + 0x290374*hit.geomUserData
                              : hit.primID);
    if (ix == dbg_x || iy == dbg_y) color = vec3f(0.f);

    if (ix == dbg_x && iy == dbg_y)
      printf("shade: Hit dist %lf idx %i\n",
             hit.t,hit.primID);
    vec4f pixel = {color.x,color.y,color.z,1.f};
    int tid = ix+iy*fbSize.x;
    d_pixels[tid] = pixel;
  }
  
  __global__
  void g_generateRays(DPRTRay *d_rays,
                      vec2i fbSize,
                      const Camera camera)
  {
    static_assert(sizeof(DPRTRay) == sizeof(Ray));
    
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;

    double u = ix+.5;
    double v = iy+.5;

    bool dbg = false; //ix == 512 && iy == 512;
    vec2d pixel = {u,v};
    Ray ray = camera.generateRay(pixel,dbg);

    // #if FORCE_BUG
    //     int dbg_x = 980; // miss
    // #else
    //     int dbg_x = 990; // hit
    // #endif
    //     int dbg_y = 500;

    //     // if (ix == dbg_x || iy == dbg_y)
    //     //   ray.tMax = 0.;
    //     if (!(ix == dbg_x && iy == dbg_y))
    //       ray.tMax = 0.;
    
    int rayID = ix+iy*fbSize.x;
    int dbg_rayID = dbg_x + dbg_y*fbSize.x;
    dbg = dbg_rayID == rayID;

    if (dbg)
      printf("ray %f %f %f : %f %f %f\n",
             (float)ray.origin.x,
             (float)ray.origin.y,
             (float)ray.origin.z,
             (float)ray.direction.x,
             (float)ray.direction.y,
             (float)ray.direction.z);
    ((Ray *)d_rays)[rayID] = ray;
  }

  void usage(const std::string &error)
  {
    if (error != "") std::cout << "Error : " << error << "\n\n";
    std::cout << "./dpMakePrimaryRays inFile.dpMini <flags>" << std::endl;
    std::cout << "/w flags:" << std::endl;
    std::cout << "  -orf outRayFile.dprays" << std::endl;
    std::cout << "  -ohf outHitFile.dphits" << std::endl;
    std::cout << "  -omf outModelFile.dpmini" << std::endl;
    std::cout << "  --ortho orthoPlaneHeight" << std::endl;
    std::cout << "  --watchDog watchDogTimeInSeconds" << std::endl;
    std::cout << "  -bc # render test-frame w/ backface culling" << std::endl;
    std::cout << "  -fc # render test-frame w/ frontface culling" << std::endl;
    std::cout << "  -oif testFrame.ppm  # where to dump test-frame to" << std::endl;
    std::cout << "  --output-res x y # res of test-frame image" << std::endl;
    exit(0);
  }

  void savePPM(std::string outImageName,
               vec2i fbSize,
               vec4f *m_pixels)
  {
    std::cout << "#dpm: writing test image to " << outImageName << std::endl;
    std::ofstream out(outImageName.c_str());
    char buf[100];
    sprintf(buf,"P3\n#deepee test image\n%i %i 255\n",fbSize.x,fbSize.y);
    out << "P3\n";
    out << "#deepeeRT test image\n";
    out << fbSize.x << " " << fbSize.y << " 255" << std::endl;
    for (int iy=0;iy<fbSize.y;iy++) {
      for (int ix=0;ix<fbSize.x;ix++) {
        vec4f pixel = m_pixels[ix+(fbSize.y-1-iy)*fbSize.x];
        auto write = [&](float f) {
          f = f*256.f;
          f = std::min(f,255.f);
          f = std::max(f,0.f);
          out << int(f) << " ";
        };
        write(pixel.x);
        write(pixel.y);
        write(pixel.z);
        out << " ";
      }
      out << "\n";
    }
  }

  void watchDog()
  {
    PING; PRINT(watchDogTime);
    if (watchDogTime == 0) return;

    PING;
    wdThread = std::thread([](){
      PING;
      int timeSlept = 0;
      while (miniapp::watchDogTime > 0) {
        if (wdTerminate) return;
        
        sleep(1);
        if (wdTerminate) return;
        
        if (++timeSlept > watchDogTime) {
          std::cout << "WATCHDOG EXPIRED!!!!" << std::endl;
          _exit(0);
        }
      }
    });
  }
  
  void main(int ac, char **av)
  {
    std::string inFileName;
    std::string outImageName = "makePrimaryRays.ppm";
    std::string outRaysName = "makePrimaryRays.dprays";
    std::string outHitsName = "makePrimaryHits.dphits";
    std::string outModelName = "";
    vec2i fbSize = { 1600,1200 };//{ 1024,1024 };
    uint64_t flags = 0;
    for (int i=1;i<ac;i++) {
      std::string arg = av[i];
      if (arg[0] != '-') {
        inFileName = arg;
      } else if (arg == "-bc" || arg == "--backface-culling") {
        flags |= DPRT_CULL_BACK;
      } else if (arg == "--watchDog") {
        watchDogTime = std::stoi(av[++i]);
      } else if (arg == "-fc" || arg == "--frontface-culling") {
        flags |= DPRT_CULL_FRONT;
      } else if (arg == "-orf" || arg == "--out-rays-file") {
        outRaysName = av[++i];
      } else if (arg == "-omf" || arg == "--out-model-file") {
        outModelName = av[++i];
      } else if (arg == "-ohf" || arg == "--out-hits-file") {
        outHitsName = av[++i];
      } else if (arg == "--native-scale") {
        view.native_scale = std::stod(av[++i]);
      } else if (arg == "--shift") {
        shift = std::stod(av[++i]);
        if (shift != 0.)
          shift = pow(10.,shift);
      } else if (arg == "-oif" || arg == "--out-image-file") {
        outImageName = av[++i];
      } else if (arg == "--camera") {
        view.from.x = std::stof(av[++i]);
        view.from.y = std::stof(av[++i]);
        view.from.z = std::stof(av[++i]);
        view.at.x   = std::stof(av[++i]);
        view.at.y   = std::stof(av[++i]);
        view.at.z   = std::stof(av[++i]);
        view.up.x   = std::stof(av[++i]);
        view.up.y   = std::stof(av[++i]);
        view.up.z   = std::stof(av[++i]);
        ++i;
        view.fovy   = std::stof(av[++i]);
      } else if (arg == "--output-res") {
        fbSize.x = std::stoi(av[++i]);
        fbSize.y = std::stoi(av[++i]);
      } else if (arg == "--ortho") {
        view.do_ortho = true;
      } else
        usage("un-recognized cmdline arg '"+arg+"'");
    }
    if (inFileName.empty())
      usage("no input file name specified");
    if (view.native_scale == 0.)
      usage("no --native-scale specified");
      
    mini::Scene::SP scene = mini::Scene::load(inFileName);

    box3d bounds = scene->getBounds();
    PRINT(bounds);
    // double scale = length(bounds.size());
    Camera camera;
    if (view.fovy == 0.) {
      view.at = bounds.center();
      view.from = view.at - length(bounds.size())*vec3d(-2,-1,-4);
      view.fovy = 60.;
    }
    PRINT(view.from);
    PRINT(view.at);
    PRINT(shift);
    
    if (view.do_ortho)
      camera = generateOrtho(fbSize,length(bounds.size())+shift);
    else
      camera = generateCamera(fbSize,shift);
    
    vec2i bs(16,16);
    vec2i nb = divRoundUp(fbSize,bs);
    
    std::cout << "#dpm: creating dprt context" << std::endl;
    DPRTContext dprt = dprtContextCreate(DPRT_CONTEXT_GPU,0);
    std::cout << "#dpm: creating model" << std::endl;
    double modelShift = 0.f;
    if (view.do_ortho)
      modelShift = 0.;
    else
      modelShift = shift;

    std::vector<mini::Mesh::SP> linearMeshes;
    DPRTModel model = toDPRT(dprt,scene,linearMeshes,
                             modelShift);

    if (outModelName != "")
      scene->save(outModelName);
    
    CUDA_SYNC_CHECK();
    size_t nRays = fbSize.x*fbSize.y;
    DPRTRay *d_rays = 0;
    cudaMalloc((void **)&d_rays,nRays*sizeof(DPRTRay));
    CUDA_SYNC_CHECK();

#if 1
    dprt_dbg_rayID = dbg_x + dbg_y*fbSize.x;
#endif
      
    g_generateRays<<<nb,bs>>>(d_rays,fbSize,camera);
    CUDA_SYNC_CHECK();


    std::vector<DPRTRay> h_rays(nRays);
    cudaMemcpy(h_rays.data(),d_rays,nRays*sizeof(DPRTRay),cudaMemcpyDefault);
    CUDA_SYNC_CHECK();
    std::ofstream f_rays(outRaysName.c_str(),std::ios::binary);
    f_rays.write((char*)&nRays,sizeof(nRays));
    PRINT(nRays);
    f_rays.write((char*)h_rays.data(),nRays*sizeof(DPRTRay));
  
    DPRTHit *d_hits = 0;
    cudaMalloc((void **)&d_hits,fbSize.x*fbSize.y*sizeof(DPRTHit));

    watchDog();
    
    CUDA_SYNC_CHECK();
    std::cout << "#dpm: calling trace" << std::endl;
    dprtTrace(model,d_rays,d_hits,fbSize.x*fbSize.y,flags);

    std::cout << "#dpm: shading rays" << std::endl;
    vec4f *m_pixels = 0;
    cudaMallocManaged((void **)&m_pixels,nRays*sizeof(vec4f));
    g_shadeRays<<<nb,bs>>>(m_pixels,d_rays,d_hits,fbSize);
    cudaStreamSynchronize(0);

    int nHits = nRays;
    std::vector<DPRTHit> h_hits(nHits);
    cudaMemcpy(h_hits.data(),d_hits,nHits*sizeof(DPRTHit),cudaMemcpyDefault);
    CUDA_SYNC_CHECK();
    if (outHitsName != "") {
      std::ofstream f_hits(outHitsName.c_str(),std::ios::binary);
      f_hits.write((char*)&nHits,sizeof(nHits));
      PRINT(nHits);
      f_hits.write((char*)h_hits.data(),nHits*sizeof(DPRTHit));
    }

    savePPM(outImageName,fbSize,m_pixels);


    shadePretty(m_pixels,fbSize,
                h_rays.data(),h_hits.data(),
                scene,linearMeshes);
    savePPM(outImageName+"_pretty.ppm",fbSize,m_pixels);
    
  }
}

int main(int ac, char **av)
{
  miniapp::main(ac,av);
  if (miniapp::watchDogTime > 0) {
    miniapp::wdTerminate = true;
    miniapp::wdThread.join();
  }
  return 0;
}
