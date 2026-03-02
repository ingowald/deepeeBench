// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dprt/dprt.h"
#include "miniScene/Scene.h"
#include <fstream>

#define CUDA_SYNC_CHECK()                                 \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      throw std::runtime_error("fatal cuda error");             \
    }                                                           \
  }


namespace miniapp {
  using namespace mini;

  /*! this HAS to be the same data layout as DPRRay in deepee.h */
  struct Ray {
    vec3d origin;
    double  tMax;
    vec3d direction;
    double  tMin;
  };

  struct Camera {
    /*! generate ray for a given pixel/image plane coordinate. based on
      how the camera was created this could be either a orthogonal or
      a perspective camera (see Camera.cpp) */
    inline __device__
    Ray generateRay(vec2d pixel, bool dbg=false) const;

    struct {
      vec3d v,du,dv;
    } origin, direction;
  };

  Camera generateCamera(vec2i imageRes,
                        const box3d &bounds,
                        const vec3d &from_dir,
                        const vec3d &up)
  {
    Camera camera;
    vec3d target = bounds.center();
    vec3d from = target + from_dir;
    vec3d direction = target-from;
    
    vec3d du = normalize(cross(direction,up));
    vec3d dv = normalize(cross(du,direction));
    
    double aspect = imageRes.x/double(imageRes.y);
    double scale = length(bounds.size());

    dv *= scale;
    du *= scale*aspect;

#if 0
    // for testing: this is a ortho camera with parallel rays and
    // different origins all n a plane
    camera.direction.v = normalize(direction);
    camera.direction.du = 0.;
    camera.direction.dv = 0.;

    camera.origin.v = from-.5*du-.5*dv;
    camera.origin.du = du * (1./imageRes.x);
    camera.origin.dv = dv * (1./imageRes.y);
#else
    // for testing: this is a perspective camera with all origins on a
    // point, and different ray directions each
    camera.direction.v = direction-.5*du-.5*dv;
    camera.direction.du = du * (1./imageRes.x);
    camera.direction.dv = dv * (1./imageRes.y);

    camera.origin.v = from;
    camera.origin.du = 0.;
    camera.origin.dv = 0.;
#endif
    return camera;
  }


  /*! generate ray for a given pixel/image plane coordinate. based on
      how the camera was created this could be either a orthogonal or
      a perspective camera (see Camera.cpp) */
  inline __device__
  Ray Camera::generateRay(vec2d pixel, bool dbg) const
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

  
  // DPRTModel createModel(DPRTContext context,
  //                       mini::Scene::SP scene)
  // {
    // std::map<dgef::Object *, DPRTGroup> objects;
    // CUDA_SYNC_CHECK();

    // for (auto inst : scene->instances)
    //   objects[inst->object] = 0;

    // std::cout << "#dpm: creating " << objects.size() << " objects" << std::endl;
    // int meshID = 0;
    // for (auto &pairs : objects) {
    //   auto obj = pairs.first;
    //   std::vector<DPRTTriangles> geoms;
    //   for (auto pm : obj->meshes) {
    //     std::cout << "#dpm: creating dprt triangle mesh w/ "
    //               << prettyNumber(pm->indices.size()) << " triangles"
    //               << std::endl;
    //     DPRTTriangles geom
    //       = dprtCreateTriangles(context,
    //                              meshID++,
    //                              (DPRTvec3*)pm->vertices.data(),
    //                              pm->vertices.size(),
    //                              (DPRTint3*)pm->indices.data(),
    //                              pm->indices.size());
    //     CUDA_SYNC_CHECK();
    //     geoms.push_back(geom);
    //   }
    //   CUDA_SYNC_CHECK();
      
    //   DPRTGroup group = dprtCreateTrianglesGroup(context,
    //                                            geoms.data(),
    //                                            geoms.size());
    //   objects[obj] = group;
    // }
    // CUDA_SYNC_CHECK();
    
    // std::cout << "#dpm: creating dprt model" << std::endl;
    // std::vector<affine3d> xfms;
    // std::vector<DPRTGroup> groups;
    // for (auto inst : scene->instances) {
    //   xfms.push_back(inst->xfm);
    //   groups.push_back(objects[inst->object]);
    // }
    // DPRTModel model = dprtCreateModel(context,
    //                                   groups.data(),
    //                                   (DPRTAffine*)xfms.data(),
    //                                   groups.size());
    // CUDA_SYNC_CHECK();
    // return model;
  // }


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

    //Ray ray = (const Ray &)d_rays[ix+iy*fbSize.x];
    DPRTHit hit = d_hits[ix+iy*fbSize.x];
    vec3f color = randomColor(hit.primID + 0x290374*hit.geomUserData);
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

    bool dbg = false;//ix == 512 && iy == 512;
    vec2d pixel = {u,v};
    Ray ray = camera.generateRay(pixel,dbg);

    int rayID = ix+iy*fbSize.x;
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
  
  void main(int ac, char **av)
  {
    std::string inFileName;
    std::string outFileName = "deepeeTest.ppm";
    vec2i fbSize = { 1024,1024 };
    uint64_t flags = 0;
    for (int i=1;i<ac;i++) {
      std::string arg = av[i];
      if (arg[0] != '-') {
        inFileName = arg;
      } else if (arg == "-bc" || arg == "--backface-culling") {
        flags |= DPRT_CULL_BACK;
      } else if (arg == "-fc" || arg == "--frontface-culling") {
        flags |= DPRT_CULL_FRONT;
      } else if (arg == "-or" || arg == "--output-res") {
        fbSize.x = std::stoi(av[++i]);
        fbSize.y = std::stoi(av[++i]);
      } else
        throw std::runtime_error("un-recognized cmdline arg '"+arg+"'");
    }
    if (inFileName.empty())
      throw std::runtime_error("no input file name specified");

    mini::Scene::SP scene = mini::Scene::load(inFileName);

    box3d bounds = scene->getBounds();
    Camera camera = generateCamera(fbSize,
                                   /* bounds to focus on */
                                   bounds,
                                   /* point we're looking from*/
                                   length(bounds.size())*vec3d(2,1,4),
                                   /* up for orientation */
                                   vec3d(0,1,0));

    vec2i bs(16,16);
    vec2i nb = divRoundUp(fbSize,bs);
    
    std::cout << "#dpm: creating dprt context" << std::endl;
    DPRTContext dprt = dprtContextCreate(DPRT_CONTEXT_GPU,0);
    std::cout << "#dpm: creating model" << std::endl;
    DPRTModel model = toDPRT(dprt,scene);

    CUDA_SYNC_CHECK();
    DPRTRay *d_rays = 0;
    cudaMalloc((void **)&d_rays,fbSize.x*fbSize.y*sizeof(DPRTRay));
    CUDA_SYNC_CHECK();
    g_generateRays<<<nb,bs>>>(d_rays,fbSize,camera);
    CUDA_SYNC_CHECK();
      
    DPRTHit *d_hits = 0;
    cudaMalloc((void **)&d_hits,fbSize.x*fbSize.y*sizeof(DPRTHit));

    CUDA_SYNC_CHECK();
    std::cout << "#dpm: calling trace" << std::endl;
    dprtTrace(model,d_rays,d_hits,fbSize.x*fbSize.y,flags);

    std::cout << "#dpm: shading rays" << std::endl;
    vec4f *m_pixels = 0;
    cudaMallocManaged((void **)&m_pixels,fbSize.x*fbSize.y*sizeof(vec4f));
    g_shadeRays<<<nb,bs>>>(m_pixels,d_rays,d_hits,fbSize);
    cudaStreamSynchronize(0);


    std::cout << "#dpm: writing test image to " << outFileName << std::endl;
    std::ofstream out(outFileName.c_str());

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
}

int main(int ac, char **av)
{
  miniapp::main(ac,av);
  return 0;
}
