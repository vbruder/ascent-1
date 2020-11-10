//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory //
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_rendering_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_rendering_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <vtkh/utils/vtkm_array_utils.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_data_object.hpp>
#include <ascent_runtime_param_check.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_web_interface.hpp> // -- for web_client_root_directory()
///
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <ascent_vtkh_collection.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkm/cont/DataSet.h>

#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#include <ascent_runtime_vtkh_utils.hpp>
#endif

#include <stdio.h>
#include <thread>

using namespace conduit;
using namespace std;

using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters::detail --
//-----------------------------------------------------------------------------
namespace detail
{
std::string
check_color_table_surprises(const conduit::Node &color_table)
{
  std::string surprises;

  std::vector<std::string> valid_paths;
  valid_paths.push_back("name");
  valid_paths.push_back("reverse");

  std::vector<std::string> ignore_paths;
  ignore_paths.push_back("control_points");

  surprises += surprise_check(valid_paths, ignore_paths, color_table);
  if(color_table.has_path("control_points"))
  {
    std::vector<std::string> c_valid_paths;
    c_valid_paths.push_back("type");
    c_valid_paths.push_back("alpha");
    c_valid_paths.push_back("color");
    c_valid_paths.push_back("position");

    const conduit::Node &control_points = color_table["control_points"];
    const int num_points = control_points.number_of_children();
    for(int i = 0; i < num_points; ++i)
    {
      const conduit::Node &point = control_points.child(i);
      surprises += surprise_check(c_valid_paths, point);
    }
  }

  return surprises;
}

std::string
check_renders_surprises(const conduit::Node &renders_node)
{
  std::string surprises;
  const int num_renders = renders_node.number_of_children();
  // render paths
  std::vector<std::string> r_valid_paths;
  r_valid_paths.push_back("image_name");
  r_valid_paths.push_back("image_prefix");
  r_valid_paths.push_back("image_width");
  r_valid_paths.push_back("image_height");
  r_valid_paths.push_back("scene_bounds");
  r_valid_paths.push_back("camera/look_at");
  r_valid_paths.push_back("camera/position");
  r_valid_paths.push_back("camera/up");
  r_valid_paths.push_back("camera/fov");
  r_valid_paths.push_back("camera/xpan");
  r_valid_paths.push_back("camera/ypan");
  r_valid_paths.push_back("camera/zoom");
  r_valid_paths.push_back("camera/near_plane");
  r_valid_paths.push_back("camera/far_plane");
  r_valid_paths.push_back("camera/azimuth");
  r_valid_paths.push_back("camera/elevation");
  r_valid_paths.push_back("type");
  r_valid_paths.push_back("phi");
  r_valid_paths.push_back("theta");
  r_valid_paths.push_back("db_name");
  r_valid_paths.push_back("render_bg");
  r_valid_paths.push_back("annotations");
  r_valid_paths.push_back("fg_color");
  r_valid_paths.push_back("bg_color");
  r_valid_paths.push_back("shading");

  for(int i = 0; i < num_renders; ++i)
  {
    const conduit::Node &render_node = renders_node.child(i);
    surprises += surprise_check(r_valid_paths, render_node);
  }
  return surprises;
}
// A simple container to create registry entries for
// renderer and the data set it renders. Without this,
// pipeline results (data sets) would be deleted before
// the Scene can be executed.
//
class RendererContainer
{
protected:
  std::string m_key;
  flow::Registry *m_registry;
  // make sure the data set we need does not get deleted
  // out from under us, which will happen
  std::shared_ptr<VTKHCollection> m_collection;
  std::string m_topo_name;
  RendererContainer() {};
public:
  RendererContainer(std::string key,
                    flow::Registry *r,
                    vtkh::Renderer *renderer,
                    std::shared_ptr<VTKHCollection> collection,
                    std::string topo_name)
    : m_key(key),
      m_registry(r),
      m_collection(collection),
      m_topo_name(topo_name)
  {
    // we have to keep around the dataset so we bring the
    // whole collection with us
    vtkh::DataSet &data = m_collection->dataset_by_topology(m_topo_name);
    renderer->SetInput(&data);
    m_registry->add<vtkh::Renderer>(m_key,renderer,1);
  }

  vtkh::Renderer *
  Fetch()
  {
    return m_registry->fetch<vtkh::Renderer>(m_key);
  }

  ~RendererContainer()
  {
    // we reset the registry in the runtime
    // which will automatically delete this pointer
    // m_registry->consume(m_key);
  }
};


class AscentScene
{
protected:
  int m_renderer_count;

  std::vector<std::vector<double> > m_render_times; // render times per renderer
  // color buffer per render per renderer
  std::vector<std::vector<std::vector<unsigned char> > > m_color_buffers;
  // distance camera position to data center per render per renderer
  std::vector<std::vector<float> > m_depths;

  flow::Registry *m_registry;
  AscentScene() {};
public:

  AscentScene(flow::Registry *r)
    : m_registry(r),
      m_renderer_count(0)
  {}

  ~AscentScene()
  {}

  std::vector<std::vector<double> > *GetRenderTimes()
  {
    return &m_render_times;
  }

  // return color buffers of all renders of selected renderer 
  std::vector<std::vector<unsigned char>> *GetColorBuffers(int rendererId)
  {
    if(rendererId >= m_renderer_count)
      ASCENT_ERROR("Trying to access data of non-existend renderer.");
    if (m_color_buffers.size() <= rendererId)
      return nullptr;

    return &m_color_buffers.at(rendererId);
  }

  // return depth of all renders of selected renderer 
  std::vector<float> *GetDepths(int rendererId)
  {
    if(rendererId >= m_renderer_count)
      ASCENT_ERROR("Trying to access data of non-existend renderer.");
    if (m_depths.size() <= rendererId)
      return nullptr;

    return &m_depths.at(rendererId);
  }
  
  int GetRendererCount()
  {
    return m_renderer_count;
  }

  void AddRenderer(RendererContainer *container)
  {
    ostringstream oss;
    oss << "key_" << m_renderer_count;
    m_registry->add<RendererContainer>(oss.str(),container,1);

    m_renderer_count++;
  }

  void Execute(std::vector<vtkh::Render> &renders, bool is_inline = false, int sleep = 0)
  {
    vtkh::Scene scene;
    for(int i = 0; i < m_renderer_count; i++)
    {
      ostringstream oss;
      oss << "key_" << i;
      vtkh::Renderer * r = m_registry->fetch<RendererContainer>(oss.str())->Fetch();
      scene.AddRenderer(r);
    }

    size_t num_renders = renders.size();
    for(size_t i = 0; i < num_renders; ++i)
    {
      scene.AddRender(renders[i]);
    }

    scene.Render(is_inline);
    std::chrono::duration<double> t_img_data;

    for(int i=0; i < m_renderer_count; i++)
    {
      int rank = 0;
#ifdef ASCENT_MPI_ENABLED
      MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
      MPI_Comm_rank(mpi_comm, &rank);
#endif
      // artificial load imbalance
      if (sleep)
      {
        // std::cout << "sleep " << sleep << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep*renders.size()));
      }

      ostringstream oss;
      oss << "key_" << i;

      if (!is_inline)
      {
        // auto start = std::chrono::system_clock::now();

        vtkh::Renderer *r = m_registry->fetch<RendererContainer>(oss.str())->Fetch();
        int size = renders.at(i).GetWidth() * renders.at(i).GetHeight();

        // move render buffers and data from vtkh to ascent
        // NOTE: only getting canvas from domain 0 for now
        // NOTE: move costs < 0.2 seconds per node per batch (200-400 renders)
        m_render_times.push_back(std::move(r->GetRenderTimes()));
        m_color_buffers.push_back(std::move(r->GetColorBuffers()));
        m_depths.push_back(std::move(r->GetDepths()));

        // t_img_data += std::chrono::system_clock::now() - start;
      }
      // std::cout << "** copy from vtkh " << t_img_data.count()  << " rank " << rank << std::endl;

      // m_registry->consume(oss.str());
    }
  }

  void ConsumeRenderers()
  {
    for (int i = 0; i < m_renderer_count; i++)
    {
      ostringstream oss;
      oss << "key_" << i;
      m_registry->consume(oss.str());
    }
  }

}; // Ascent Scene

//-----------------------------------------------------------------------------

vtkh::Render parse_render(const conduit::Node &render_node,
                          vtkm::Bounds &bounds,
                          const std::string &image_name)
{
  int image_width;
  int image_height;

  parse_image_dims(render_node, image_width, image_height);

  //
  // for now, all the canvases we support are the same
  // so passing MakeRender a RayTracer is ok
  //
  vtkh::Render render = vtkh::MakeRender(image_width,
                                         image_height,
                                         bounds,
                                         image_name);
  //
  // render create a default camera. Now get it and check for
  // values that override the default view
  //
  if(render_node.has_path("camera"))
  {
    vtkm::rendering::Camera camera = render.GetCamera();
    parse_camera(render_node["camera"], camera);
    render.SetCamera(camera);
  }
  if(render_node.has_path("shading"))
  {
    bool on = render_node["shading"].as_string() == "enabled";
    render.SetShadingOn(on);
  }

  if(render_node.has_path("annotations"))
  {
    if(!render_node["annotations"].dtype().is_string())
    {
      ASCENT_ERROR("render/annotations node must be a string value");
    }
    const std::string annot = render_node["annotations"].as_string();
    // default is always render annotations
    if(annot == "false")
    {
      render.DoRenderAnnotations(false);
    }
  }

  if(render_node.has_path("render_bg"))
  {
    if(!render_node["render_bg"].dtype().is_string())
    {
      ASCENT_ERROR("render/render_bg node must be a string value");
    }
    const std::string render_bg = render_node["render_bg"].as_string();
    // default is always render the background
    // off will make the background transparent
    if(render_bg == "false")
    {
      render.DoRenderBackground(false);
    }
  }

  if(render_node.has_path("bg_color"))
  {
    if(!render_node["bg_color"].dtype().is_number() ||
       render_node["bg_color"].dtype().number_of_elements() != 3)
    {
      ASCENT_ERROR("render/bg_color node must be an array of 3 values");
    }
    conduit::Node n;
    render_node["bg_color"].to_float32_array(n);
    const float32 *color = n.as_float32_ptr();
    float32 color4f[4];
    color4f[0] = color[0];
    color4f[1] = color[1];
    color4f[2] = color[2];
    color4f[3] = 1.f;
    render.SetBackgroundColor(color4f);
  }

  if(render_node.has_path("fg_color"))
  {
    if(!render_node["fg_color"].dtype().is_number() ||
       render_node["fg_color"].dtype().number_of_elements() != 3)
    {
      ASCENT_ERROR("render/fg_color node must be an array of 3 values");
    }
    conduit::Node n;
    render_node["fg_color"].to_float32_array(n);
    const float32 *color = n.as_float32_ptr();
    float32 color4f[4];
    color4f[0] = color[0];
    color4f[1] = color[1];
    color4f[2] = color[2];
    color4f[3] = 1.f;
    render.SetForegroundColor(color4f);
  }

  return render;
}

class CinemaManager
{
protected:
  std::vector<vtkm::rendering::Camera> m_cameras;
  std::vector<std::string>             m_image_names;
  std::vector<float>                   m_phi_values;
  std::vector<float>                   m_theta_values;
  std::vector<float>                   m_times;
  std::string                          m_csv;

  vtkm::Bounds                         m_bounds;
  const int                            m_phi;
  const int                            m_theta;
  std::string                          m_image_name;
  std::string                          m_image_path;
  std::string                          m_db_path;
  std::string                          m_base_path;
  float                                m_time;
public:
  CinemaManager(vtkm::Bounds bounds,
                const int phi,
                const int theta,
                const std::string image_name,
                const std::string path)
    : m_bounds(bounds),
      m_phi(phi),
      m_theta(theta),
      m_image_name(image_name),
      m_time(0.f)
  {
    this->create_cinema_cameras(bounds);
    m_csv = "phi,theta,time,FILE\n";

    m_base_path = conduit::utils::join_file_path(path, "cinema_databases");
  }

  CinemaManager()
    : m_phi(0),
      m_theta(0)
  {
    ASCENT_ERROR("Cannot create un-initialized CinemaManger");
  }

  void set_bounds(vtkm::Bounds &bounds)
  {
    if(bounds != m_bounds)
    {
      this->create_cinema_cameras(bounds);
    }
  }

  void add_time_step(bool is_intransit = false)
  {
    m_times.push_back(m_time);

    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    int size = 0;
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);
    // use last rank to generate dir for in transit only case
    if (is_intransit && rank == size - 1)
      rank = 0;
#endif
    if(rank == 0 && !conduit::utils::is_directory(m_base_path))
    {
        conduit::utils::create_directory(m_base_path);
    }

    // add a database path
    m_db_path = conduit::utils::join_file_path(m_base_path, m_image_name);

    if(rank == 0 && !conduit::utils::is_directory(m_db_path))
    {
        conduit::utils::create_directory(m_db_path);
        // copy over cinema web resources
        std::string cinema_root = conduit::utils::join_file_path(web_client_root_directory(),
                                                                 "cinema");
        ascent::copy_directory(cinema_root, m_db_path);
    }

    std::stringstream ss;
    ss<<fixed<<showpoint;
    ss<<std::setprecision(1)<<m_time;
    // add a time step path
    m_image_path = conduit::utils::join_file_path(m_db_path,ss.str());

    if(!conduit::utils::is_directory(m_image_path))
    {
        conduit::utils::create_directory(m_image_path);
    }

    m_time += 1.f;
  }

  void fill_renders(std::vector<vtkh::Render> *renders,
                    const conduit::Node &render_node,
                    const int current_render_count,
                    const int render_offset,
                    const bool is_probing,
                    const std::vector<int> &probing_sequence)
  {
    conduit::Node render_copy = render_node;

    // allow zoom to be ajusted
    conduit::Node zoom;
    if(render_copy.has_path("camera/zoom"))
    {
      zoom = render_copy["camera/zoom"];
    }

    // cinema is controlling the camera so get
    // rid of it
    if(render_copy.has_path("camera"))
    {
      render_copy["camera"].reset();
    }

    std::string tmp_name = "";
    vtkh::Render render = detail::parse_render(render_copy,
                                               m_bounds,
                                               tmp_name);
    int num_renders = m_image_names.size();
    // std::cout << "FILL RENDERS " << render_offset << " - " << current_render_count << std::endl;

    // adjust render count
    if (current_render_count > 0)
      num_renders = current_render_count;

    int probing_it = 0;
    while (probing_sequence.size() > probing_it && probing_sequence[probing_it] < render_offset)
      ++probing_it;

    int i = render_offset;
    if (is_probing && probing_sequence.size() > probing_it)
      i = probing_sequence[probing_it]; // first probing render

    while (i < render_offset + num_renders)
    {
      if (probing_sequence.size() <= probing_it)
      {
        ASCENT_ERROR("Missing sampling sequence in runtime rendering filters.");
        break;
      }
      if (!is_probing && (i == probing_sequence[probing_it]))
      {
        ++i;
        if (probing_sequence.size() > probing_it + 1)
          ++probing_it;
        continue;     // skip render, already rendered while probing
      }

      std::string image_name = conduit::utils::join_file_path(m_image_path , m_image_names[i]);

      render.SetImageName(image_name);
      // we have to make a copy of the camera because
      // zoom is additive for some reason
      vtkm::rendering::Camera camera = m_cameras[i];

      if(!zoom.dtype().is_empty())
      {
        // Allow default zoom to be overridden
        double vtkm_zoom = zoom_to_vtkm_zoom(zoom.to_float64());
        camera.Zoom(vtkm_zoom);
      }

      render.SetCamera(camera);
      renders->push_back(render);

      if (is_probing && probing_sequence.size() > ++probing_it)
        i = probing_sequence[probing_it];  // skip to next probing image
      else if (is_probing)
        break;    // this was the last probing image
      else  // non-probing, advance to the next image
        ++i;
    }
    // std::cout << "___renders.size " << renders->size() << std::endl;
  }

  std::string get_string(const float value)
  {
    std::stringstream ss;
    ss<<std::fixed<<std::setprecision(1)<<value;
    return ss.str();
  }

  void write_metadata()
  {
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    if(rank != 0)
    {
      return;
    }
    conduit::Node meta;
    meta["type"] = "simple";
    meta["version"] = "1.1";
    meta["metadata/type"] = "parametric-image-stack";
    meta["name_pattern"] = "{time}/{phi}_{theta}_" + m_image_name + ".png";

    conduit::Node times;
    times["default"] = get_string(m_times[0]);
    times["label"] = "time";
    times["type"] = "range";
    // we have to make sure that this maps to a json array
    const int t_size = m_times.size();
    for(int i = 0; i < t_size; ++i)
    {
      times["values"].append().set(get_string(m_times[i]));
    }

    meta["arguments/time"] = times;

    conduit::Node phis;
    phis["default"] = get_string(m_phi_values[0]);
    phis["label"] = "phi";
    phis["type"] = "range";
    const int phi_size = m_phi_values.size();
    for(int i = 0; i < phi_size; ++i)
    {
      phis["values"].append().set(get_string(m_phi_values[i]));
    }

    meta["arguments/phi"] = phis;

    conduit::Node thetas;
    thetas["default"] = get_string(m_theta_values[0]);
    thetas["label"] = "theta";
    thetas["type"] = "range";
    const int theta_size = m_theta_values.size();
    for(int i = 0; i < theta_size; ++i)
    {
      thetas["values"].append().set(get_string(m_theta_values[i]));
    }

    meta["arguments/theta"] = thetas;
    meta.save(m_db_path + "/info.json","json");

    // also generate info.js, a simple javascript variant of
    // info.json that our index.html reads directly to
    // avoid ajax

    std::ofstream out_js(m_db_path + "/info.js");
    out_js<<"var info =";
    meta.to_json_stream(out_js);
    out_js.close();

    //append current data to our csv file
    std::stringstream csv;

    csv<<m_csv;
    std::string current_time = get_string(m_times[t_size - 1]);
    for(int p = 0; p < phi_size; ++p)
    {
      std::string phi = get_string(m_phi_values[p]);
      for(int t = 0; t < theta_size; ++t)
      {
        std::string theta = get_string(m_theta_values[t]);
        csv<<phi<<",";
        csv<<theta<<",";
        csv<<current_time<<",";
        csv<<current_time<<"/"<<phi<<"_"<<theta<<"_"<<m_image_name<<".png\n";
      }
    }

    m_csv = csv.str();
    std::ofstream out(m_db_path + "/data.csv");
    out<<m_csv;
    out.close();

  }

private:
  void create_cinema_cameras(vtkm::Bounds bounds)
  {
    m_cameras.clear();
    m_image_names.clear();
    using vtkmVec3f = vtkm::Vec<vtkm::Float32,3>;
    vtkmVec3f center = bounds.Center();
    vtkm::Vec<vtkm::Float32,3> totalExtent;
    totalExtent[0] = vtkm::Float32(bounds.X.Length());
    totalExtent[1] = vtkm::Float32(bounds.Y.Length());
    totalExtent[2] = vtkm::Float32(bounds.Z.Length());

    vtkm::Float32 radius = vtkm::Magnitude(totalExtent) * 2.5 / 2.0;

    const double pi = 3.141592653589793;
    double phi_inc = 360.0 / double(m_phi);
    double theta_inc = 180.0 / double(m_theta);
    for(int p = 0; p < m_phi; ++p)
    {
      float phi  =  -180.f + phi_inc * p;
      m_phi_values.push_back(phi);

      for(int t = 0; t < m_theta; ++t)
      {
        float theta = theta_inc * t;
        if (p == 0)
        {
          m_theta_values.push_back(theta);
        }

        const int i = p * m_theta + t;

        vtkm::rendering::Camera camera;
        camera.ResetToBounds(bounds);

        //
        //  spherical coords start (r=1, theta = 0, phi = 0)
        //  (x = 0, y = 0, z = 1)
        //

        vtkmVec3f pos(0.f,0.f,1.f);
        vtkmVec3f up(0.f,1.f,0.f);

        vtkm::Matrix<vtkm::Float32,4,4> phi_rot;
        vtkm::Matrix<vtkm::Float32,4,4> theta_rot;
        vtkm::Matrix<vtkm::Float32,4,4> rot;

        phi_rot = vtkm::Transform3DRotateZ(phi);
        theta_rot = vtkm::Transform3DRotateX(theta);
        rot = vtkm::MatrixMultiply(phi_rot, theta_rot);

        up = vtkm::Transform3DVector(rot, up);
        vtkm::Normalize(up);

        pos = vtkm::Transform3DPoint(rot, pos);
        pos = pos * radius + center;

        camera.SetViewUp(up);
        camera.SetLookAt(center);
        camera.SetPosition(pos);
        //camera.Zoom(0.2f);

        std::stringstream ss;
        ss<<get_string(phi)<<"_"<<get_string(theta)<<"_";

        m_image_names.push_back(ss.str() + m_image_name);
        m_cameras.push_back(camera);

      } // theta
    } // phi
  }

}; // CinemaManager

class CinemaDatabases
{
private:
  static std::map<std::string, CinemaManager> m_databases;
public:

  static bool db_exists(std::string db_name)
  {
    auto it = m_databases.find(db_name);
    return it != m_databases.end();
  }

  static void create_db(vtkm::Bounds bounds,
                        const int phi,
                        const int theta,
                        std::string db_name,
                        std::string path)
  {
    if(db_exists(db_name))
    {
      ASCENT_ERROR("Creation failed: cinema database already exists");
    }

    m_databases.emplace(std::make_pair(db_name, CinemaManager(bounds, phi, theta, db_name, path)));
  }

  static CinemaManager& get_db(std::string db_name)
  {
    if(!db_exists(db_name))
    {
      ASCENT_ERROR("Cinema db '"<<db_name<<"' does not exist.");
    }

    return m_databases[db_name];
  }
};

std::map<std::string, CinemaManager> CinemaDatabases::m_databases;

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end namespace detail --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
DefaultRender::DefaultRender()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DefaultRender::~DefaultRender()
{
// empty
}

//-----------------------------------------------------------------------------
void
DefaultRender::declare_interface(Node &i)
{
    i["type_name"] = "default_render";
    i["port_names"].append() = "a";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
DefaultRender::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("image_name",params, info, false);
    res &= check_string("image_prefix",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("image_prefix");
    valid_paths.push_back("image_name");

    std::vector<std::string> ignore_paths;
    ignore_paths.push_back("renders");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);


    // parse render surprises
    if(params.has_path("renders"))
    {
      const conduit::Node &renders_node = params["renders"];
      surprises += detail::check_renders_surprises(renders_node);
    }

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------

void
DefaultRender::execute()
{
    if(!input(0).check_type<vtkm::Bounds>())
    {
      ASCENT_ERROR("'a' input must be a vktm::Bounds * instance");
    }

    vtkm::Bounds *bounds = input<vtkm::Bounds>(0);

    std::vector<vtkh::Render> *renders = new std::vector<vtkh::Render>();

    Node * meta = graph().workspace().registry().fetch<Node>("metadata");

    int cycle = 0;

    if(meta->has_path("cycle"))
    {
      cycle = (*meta)["cycle"].as_int32();
    }

    if(params().has_path("renders"))
    {
      const conduit::Node renders_node = params()["renders"];
      const int num_renders = renders_node.number_of_children();

      for(int i = 0; i < num_renders; ++i)
      {
        const conduit::Node render_node = renders_node.child(i);
        std::string image_name;

        bool is_cinema = false;

        if(render_node.has_path("type"))
        {
          if(render_node["type"].as_string() == "cinema")
          {
            is_cinema = true;
          }
        }

        if(is_cinema)
        {
          int phi = 5;
          int theta = 5;
          if (render_node.has_path("phi"))
            phi = render_node["phi"].to_int32();
          if (render_node.has_path("theta"))
            theta = render_node["theta"].to_int32();

          const int full_render_count = phi*theta;
          int current_render_count = full_render_count;
          int render_offset = 0;
          
          if (meta->has_path("render_count"))
          {
            if ((*meta)["render_count"].as_int32() > 0)
            {
              current_render_count = (*meta)["render_count"].as_int32();
            }
          }
          if (meta->has_path("render_offset"))
          {
            render_offset = (*meta)["render_offset"].as_int32();
          }

          // check if probing run
          double probing_factor = 0.0;
          int stride = 1;

          std::vector<int> probing_sequence;
          bool is_probing = false;
          if (meta->has_path("is_probing") && meta->has_path("probing_factor"))
          {
            probing_factor = (*meta)["probing_factor"].as_double();
            if (probing_factor > 0.0)
            {
              stride = int(std::round(full_render_count / (probing_factor*full_render_count)));
              if ((*meta)["is_probing"].as_int32())
                is_probing = true;
            }

            std::string sampling_method;
            if (meta->has_path("sampling_method"))
            {
              sampling_method = (*meta)["sampling_method"].as_string();
              if (sampling_method == "random")
              {
                std::srand(42);
                const int range_from  = 0;
                const int range_to    = full_render_count;
                const int probing_count = int(probing_factor * full_render_count);
                probing_sequence.resize(probing_count);
                for (int i = 0; i < probing_count; ++i)
                {
                  int r = (double(std::rand()) / double(RAND_MAX - 1)) * (range_to - range_from + 1) + range_from;
                  probing_sequence[i] = r;
                }
                std::sort(probing_sequence.begin(), probing_sequence.end());
              }
              else if (stride > 0)  // systematic
              {
                int pos = 0;
                do 
                {
                  probing_sequence.push_back(pos);
                  pos += stride;
                } while (pos < full_render_count);
              }
              
              // std::cout << "probing sequence: ";
              // for (auto &a : probing_sequence)
              //     std::cout << a << " ";
              // std::cout << std::endl;
            }
          }

          bool is_cinema_increment = false;
          if (meta->has_path("cinema_increment"))
            is_cinema_increment = (*meta)["cinema_increment"].as_int32();
          std::string insitu_type;
          if (meta->has_path("insitu_type"))
            insitu_type = (*meta)["insitu_type"].as_string();

          std::string output_path = default_dir(graph());
          if (render_node.has_path("output_path"))
          {
            output_path = render_node["output_path"].as_string();
          }

          std::string db_name = "cinema_db";
          if (render_node.has_path("db_name"))
          {
            db_name = render_node["db_name"].as_string();
          }
          else
          {
            ASCENT_INFO("No cinema 'db_name' specified, defaulting to 'cinema_db'.");
          }

          bool exists = detail::CinemaDatabases::db_exists(db_name);
          if(!exists)
          {
            detail::CinemaDatabases::create_db(*bounds,phi,theta, db_name, output_path);
          }
          detail::CinemaManager &manager = detail::CinemaDatabases::get_db(db_name);

          int image_width;
          int image_height;
          parse_image_dims(render_node, image_width, image_height);

          manager.set_bounds(*bounds);
          // Add new timestep only for probing runs, otherwise we generate too many.
          if (is_probing || (!is_probing && is_cinema_increment)) 
          {
            manager.add_time_step(insitu_type == "intransit");
          }
          manager.fill_renders(renders, render_node, current_render_count, render_offset, 
                               is_probing, probing_sequence);
          manager.write_metadata();
        }
        else
        {
          // this render has a unique name
          if(render_node.has_path("image_name"))
          {
            image_name = render_node["image_name"].as_string();
            image_name = output_dir(image_name, graph());
          }
          else if(render_node.has_path("image_prefix"))
          {
            std::stringstream ss;
            ss<<expand_family_name(render_node["image_prefix"].as_string(), cycle);
            image_name = ss.str();
            image_name = output_dir(image_name, graph());
          }
          else
          {
            std::string render_name = renders_node.child_names()[i];
            std::string fpath = filter_to_path(this->name());
            ASCENT_ERROR("Render ("<<fpath<<"/"<<render_name<<")"<<
                         " must have either a 'image_name' or "
                         "'image_prefix' parameter");
          }

          vtkh::Render render = detail::parse_render(render_node,
                                                     *bounds,
                                                     image_name);
          renders->push_back(render);
        }
      }
    }
    else
    {
      // This is the path for the default render attached directly to a scene
      std::string image_name;
      if(params().has_path("image_name"))
      {
        image_name =  params()["image_name"].as_string();
      }
      else
      {
        image_name =  params()["image_prefix"].as_string();
        image_name = expand_family_name(image_name, cycle);
      }

      vtkh::Render render = vtkh::MakeRender(1024,
                                             1024,
                                             *bounds,
                                             image_name);

      renders->push_back(render);
    }
    set_output<std::vector<vtkh::Render>>(renders);
}

//-----------------------------------------------------------------------------
VTKHBounds::VTKHBounds()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHBounds::~VTKHBounds()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHBounds::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_bounds";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
VTKHBounds::execute()
{
    vtkm::Bounds *bounds = new vtkm::Bounds;

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHBounds input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    // std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    // bounds->Include(collection->global_bounds());

    // TODO: calculate bounds w/o global sync
    bounds->X.Min =  0.0;
    bounds->X.Max = 10.0;
    bounds->Y.Min =  0.0;
    bounds->Y.Max = 10.0;
    bounds->Z.Min =  0.0;
    bounds->Z.Max = 10.0;

    set_output<vtkm::Bounds>(bounds);
}


//-----------------------------------------------------------------------------
VTKHUnionBounds::VTKHUnionBounds()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHUnionBounds::~VTKHUnionBounds()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHUnionBounds::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_union_bounds";
    i["port_names"].append() = "a";
    i["port_names"].append() = "b";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
VTKHUnionBounds::execute()
{
    if(!input(0).check_type<vtkm::Bounds>())
    {
        ASCENT_ERROR("'a' must be a vtkm::Bounds * instance");
    }

    if(!input(1).check_type<vtkm::Bounds>())
    {
        ASCENT_ERROR("'b' must be a vtkm::Bounds * instance");
    }

    vtkm::Bounds *result = new vtkm::Bounds;

    vtkm::Bounds *bounds_a = input<vtkm::Bounds>(0);
    vtkm::Bounds *bounds_b = input<vtkm::Bounds>(1);

    result->Include(*bounds_a);
    result->Include(*bounds_b);
    set_output<vtkm::Bounds>(result);
}

//-----------------------------------------------------------------------------
AddPlot::AddPlot()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
AddPlot::~AddPlot()
{
// empty
}

//-----------------------------------------------------------------------------
void
AddPlot::declare_interface(Node &i)
{
    i["type_name"] = "add_plot";
    i["port_names"].append() = "scene";
    i["port_names"].append() = "plot";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
void
AddPlot::execute()
{
    if(!input(0).check_type<detail::AscentScene>())
    {
        ASCENT_ERROR("'scene' must be a AscentScene * instance");
    }

    if(!input(1).check_type<detail::RendererContainer >())
    {
        ASCENT_ERROR("'plot' must be a detail::RendererContainer * instance");
    }

    detail::AscentScene *scene = input<detail::AscentScene>(0);
    detail::RendererContainer * cont = input<detail::RendererContainer>(1);
    scene->AddRenderer(cont);
    set_output<detail::AscentScene>(scene);
}

//-----------------------------------------------------------------------------
CreatePlot::CreatePlot()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
CreatePlot::~CreatePlot()
{
// empty
}

//-----------------------------------------------------------------------------
void
CreatePlot::declare_interface(Node &i)
{
    i["type_name"] = "create_plot";
    i["port_names"].append() = "a";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
bool
CreatePlot::verify_params(const conduit::Node &params,
                          conduit::Node &info)
{
    info.reset();

    bool res = check_string("type",params, info, true);

    bool is_mesh = false;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("type");
    valid_paths.push_back("pipeline");

    res &= check_string("topology",params, info, false);
    valid_paths.push_back("topology");

    if(res)
   {
      if(params["type"].as_string() == "mesh")
      {
        is_mesh = true;
      }
    }

    if(!is_mesh)
    {
      res &= check_string("field", params, info, true);
      valid_paths.push_back("field");
      valid_paths.push_back("points/radius");
      valid_paths.push_back("points/radius_delta");
      valid_paths.push_back("min_value");
      valid_paths.push_back("max_value");
      valid_paths.push_back("samples");
    }
    else
    {
      valid_paths.push_back("overlay");
      valid_paths.push_back("show_internal");
    }


    std::vector<std::string> ignore_paths;
    ignore_paths.push_back("color_table");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(params.has_path("color_table"))
    {
      surprises += detail::check_color_table_surprises(params["color_table"]);
    }

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
CreatePlot::execute()
{
    if(!input(0).check_type<DataObject>())
    {
      ASCENT_ERROR("create_plot input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    conduit::Node &plot_params = params();
    std::string field_name;
    if(plot_params.has_path("field"))
    {
      field_name = plot_params["field"].as_string();
    }
    std::string topo_name;
    if(field_name == "")
    {
      topo_name = detail::resolve_topology(params(),
                                           this->name(),
                                           collection);
    }
    else
    {
      topo_name = collection->field_topology(field_name);
      if(topo_name == "")
      {
        detail::field_error(field_name, this->name(), collection);
      }
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    std::string type = params()["type"].as_string();

    // TODO: avoid global check
    // if(data.GlobalIsEmpty())
    // {
    //   std::string fpath = filter_to_path(this->name());
    //   ASCENT_INFO(fpath<<" "<<type<<" plot yielded no data, i.e., no cells remain");
    // }

    vtkh::Renderer *renderer = nullptr;

    if(type == "pseudocolor")
    {
      bool is_point_mesh = data.IsPointMesh();
      if(is_point_mesh)
      {
        vtkh::PointRenderer *p_renderer = new vtkh::PointRenderer();
        p_renderer->UseCells();
        if(plot_params.has_path("points/radius"))
        {
          float radius = plot_params["points/radius"].to_float32();
          p_renderer->SetBaseRadius(radius);
        }
        // default is to use a constant radius
        // if the radius delta is present, we will
        // vary radii based on the scalar value
        if(plot_params.has_path("points/radius_delta"))
        {
          float radius = plot_params["points/radius_delta"].to_float32();
          p_renderer->UseVariableRadius(true);
          p_renderer->SetRadiusDelta(radius);
        }
        renderer = p_renderer;
      }
      else
      {
        renderer = new vtkh::RayTracer();
      }

    }
    else if(type == "volume")
    {
      vtkh::VolumeRenderer *vren = new vtkh::VolumeRenderer();
      if(plot_params.has_path("samples"))
      {
        int samples = plot_params["samples"].to_int32();
        vren->SetNumberOfSamples(samples);
      }
      renderer = vren;
    }
    else if(type == "mesh")
    {
      renderer = new vtkh::MeshRenderer();
    }
    else
    {
        ASCENT_ERROR("create_plot unknown plot type '"<<type<<"'");
    }

    // get the plot params
    if(plot_params.has_path("color_table"))
    {
      vtkm::cont::ColorTable color_table =  parse_color_table(plot_params["color_table"]);
      renderer->SetColorTable(color_table);
    }

    vtkm::Range scalar_range;
    if(plot_params.has_path("min_value"))
    {
      scalar_range.Min = plot_params["min_value"].to_float64();
    }

    if(plot_params.has_path("max_value"))
    {
      scalar_range.Max = plot_params["max_value"].to_float64();
    }

    renderer->SetRange(scalar_range);

    if(field_name != "")
    {
      renderer->SetField(field_name);
    }

    if(type == "mesh")
    {
      vtkh::MeshRenderer *mesh = dynamic_cast<vtkh::MeshRenderer*>(renderer);
      if(!plot_params.has_path("field"))
      {
        // The renderer needs a field, so add one if
        // needed. This will eventually go away once
        // the mesh mapper in vtkm can handle no field
        const std::string fname = "constant_mesh_field";
        data.AddConstantPointField(0.f, fname);
        renderer->SetField(fname);
        mesh->SetUseForegroundColor(true);
      }

      mesh->SetIsOverlay(true);
      if(plot_params.has_path("overlay"))
      {
        if(plot_params["overlay"].as_string() == "false")
        {
          mesh->SetIsOverlay(false);
        }
      }

      if(plot_params.has_path("show_internal"))
      {
        if(plot_params["show_internal"].as_string() == "true")
        {
          mesh->SetShowInternal(true);
        }
      }
    } // is mesh

    std::string key = this->name() + "_cont";

    detail::RendererContainer *container = new detail::RendererContainer(key,
                                                                         &graph().workspace().registry(),
                                                                         renderer,
                                                                         collection,
                                                                         topo_name);
    set_output<detail::RendererContainer>(container);

}


//-----------------------------------------------------------------------------
CreateScene::CreateScene()
: Filter()
{}

//-----------------------------------------------------------------------------
CreateScene::~CreateScene()
{}

//-----------------------------------------------------------------------------
void
CreateScene::declare_interface(Node &i)
{
    i["type_name"]   = "create_scene";
    i["output_port"] = "true";
    i["port_names"] = DataType::empty();
}

//-----------------------------------------------------------------------------
void
CreateScene::execute()
{
    detail::AscentScene *scene = new detail::AscentScene(&graph().workspace().registry());
    set_output<detail::AscentScene>(scene);
}


void add_images(std::vector<vtkh::Render> *renders, 
                flow::Graph *graph, 
                const std::vector<std::vector<double> > *scene_render_times,
                std::vector<std::vector<unsigned char> > *color_buffers,
                std::vector<float> *depths)
{
  // check if anything was rendered
  if (color_buffers->size() == 0)
  {
    std::cout << "no image to add." << std::endl;
    return;
  }

  if (!graph->workspace().registry().has_entry("image_list"))
  {
    conduit::Node *image_list = new conduit::Node();
    graph->workspace().registry().add<Node>("image_list", image_list, 1);
  }
  conduit::Node *image_list = graph->workspace().registry().fetch<Node>("image_list");

  auto start = std::chrono::system_clock::now();

  std::vector<conduit::Node> image_data(renders->size());

  for (int i = 0; i < renders->size(); ++i)
    image_list->append();

#pragma omp parallel for
  for (int i = 0; i < renders->size(); ++i)
  {
    const std::string image_name = renders->at(i).GetImageName() + ".png";

    image_data.at(i)["image_name"] = image_name;
    image_data[i]["image_width"] = renders->at(i).GetWidth();
    image_data[i]["image_height"] = renders->at(i).GetHeight();

    image_data[i]["camera/position"].set(&renders->at(i).GetCamera().GetPosition()[0], 3);
    image_data[i]["camera/look_at"].set(&renders->at(i).GetCamera().GetLookAt()[0], 3);
    image_data[i]["camera/up"].set(&renders->at(i).GetCamera().GetViewUp()[0], 3);
    image_data[i]["camera/zoom"] = renders->at(i).GetCamera().GetZoom();
    image_data[i]["camera/fov"] = renders->at(i).GetCamera().GetFieldOfView();
    vtkm::Bounds bounds = renders->at(i).GetSceneBounds();
    double coord_bounds[6] = {bounds.X.Min,
                              bounds.Y.Min,
                              bounds.Z.Min,
                              bounds.X.Max,
                              bounds.Y.Max,
                              bounds.Z.Max};
    image_data[i]["scene_bounds"].set(coord_bounds, 6);

    double avg_render_time = 0.0;
    int count = 0;
    // loop over renderers
    for (size_t j = 0; j < scene_render_times->size(); ++j)
    {
      // NOTE: average over render times for now
      if (scene_render_times->at(j).size() > i)
      {
        avg_render_time += scene_render_times->at(j).at(i);
        ++count;
      }
    }

    avg_render_time /= count ? double(count) : 1.0;
    image_data[i]["render_time"] = avg_render_time;

    int size = renders->at(i).GetWidth() * renders->at(i).GetHeight();
    // NOTE: only getting canvas from domain 0 for now
    image_data[i]["color_buffer"].set_external(color_buffers->at(i).data(), size * 4); // *4 for RGBA

    // image_data[i]["depth_buffer"].set_external(depth_buffers->at(i).data(), size);
    image_data[i]["depth"] = depths->at(i);

    // TODO: copy: big performance hit -> avoid copy of color buffer and move uchar conversion
    // set_external is way faster (no copy) but results in empty packed messages (png write)
    // image_list->child(i).set_external(image_data[i]);

    // Node &image = image_list->append();
    // image.set(std::move(image_data[i]));

    image_list->child(i).set(std::move(image_data[i]));

    float* depth_buffer = vtkh::GetVTKMPointer(renders->at(i).GetCanvas().GetDepthBuffer());
    image_list->child(i)["depth_buffer"].set_external(depth_buffer, size);
    // float* color_buffer = &vtkh::GetVTKMPointer(renders->at(i).GetCanvas()->GetColorBuffer())[0][0];
    // image_list->child(i)["color_buffer"].set_external(color_buffer, size * 4);
    
    // image_list->append() = image_data;

    // append name and frame time to ascent info
    // conduit::Node image_info;
    // image_info["image_name"] = image_name;
    // image_info["render_times"] = render_times;
    // info["renders"].append() = image_info;
  } // for renders

  std::chrono::duration<double> t_buffers = std::chrono::system_clock::now() - start;
}


//-----------------------------------------------------------------------------
ExecScene::ExecScene()
  : Filter()
{

}

//-----------------------------------------------------------------------------
ExecScene::~ExecScene()
{

}

//-----------------------------------------------------------------------------
void
ExecScene::declare_interface(conduit::Node &i)
{
    i["type_name"] = "exec_scene";
    i["port_names"].append() = "scene";
    i["port_names"].append() = "renders";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
void
ExecScene::execute()
{
    if(!input(0).check_type<detail::AscentScene>())
    {
        ASCENT_ERROR("'scene' must be a AscentScene * instance");
    }

    if(!input(1).check_type<std::vector<vtkh::Render> >())
    {
        ASCENT_ERROR("'renders' must be a std::vector<vtkh::Render> * instance");
    }

    detail::AscentScene *scene = input<detail::AscentScene>(0);
    std::vector<vtkh::Render> * renders = input<std::vector<vtkh::Render>>(1);

    bool is_inline = false;
    Node *meta = graph().workspace().registry().fetch<Node>("metadata");
    if (meta->has_path("insitu_type"))
      is_inline = (*meta)["insitu_type"].as_string() == "inline";
    int sleep = 0;
    if (meta->has_path("sleep"))
      sleep = (*meta)["sleep"].as_int32();

    scene->Execute(*renders, is_inline, sleep);

    std::vector<std::vector<double> > *render_times = scene->GetRenderTimes();
    // NOTE: only domain 0 for now
    std::vector<std::vector<unsigned char> > *color_buffers = scene->GetColorBuffers(0);
    std::vector<float> *depths = scene->GetDepths(0);

    if (!is_inline)
      add_images(renders, &graph(), render_times, color_buffers, depths);

    // the images should exist now so add them to the image list
    // this can be used for the web server or jupyter

    // if(!graph().workspace().registry().has_entry("image_list"))
    // {
    //   conduit::Node *image_list = new conduit::Node();
    //   graph().workspace().registry().add<Node>("image_list", image_list,1);
    // }

    // conduit::Node *image_list = graph().workspace().registry().fetch<Node>("image_list");
    // for(int i = 0; i < renders->size(); ++i)
    // {
    //   const std::string image_name = renders->at(i).GetImageName() + ".png";
    //   conduit::Node image_data;
    //   image_data["image_name"] = image_name;
    //   image_data["image_width"] = renders->at(i).GetWidth();
    //   image_data["image_height"] = renders->at(i).GetHeight();

    //   image_data["camera/position"].set(&renders->at(i).GetCamera().GetPosition()[0],3);
    //   image_data["camera/look_at"].set(&renders->at(i).GetCamera().GetLookAt()[0],3);
    //   image_data["camera/up"].set(&renders->at(i).GetCamera().GetViewUp()[0],3);
    //   image_data["camera/zoom"] = renders->at(i).GetCamera().GetZoom();
    //   image_data["camera/fov"] = renders->at(i).GetCamera().GetFieldOfView();
    //   vtkm::Bounds bounds=  renders->at(i).GetSceneBounds();
    //   double coord_bounds [6] = {bounds.X.Min,
    //                              bounds.Y.Min,
    //                              bounds.Z.Min,
    //                              bounds.X.Max,
    //                              bounds.Y.Max,
    //                              bounds.Z.Max};

    //   image_data["scene_bounds"].set(coord_bounds, 6);

    //   image_list->append() = image_data;
    // }

    scene->ConsumeRenderers();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
