#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <boost/numpy.hpp>
#include <iostream>

#include "GL/osmesa.h"
#include <GL/gl.h> 
#include <GL/glu.h> 
#include <GL/freeglut.h> 
#define GLAPIENTRY

// global rending constants
float near = 0.05f;
float far = 1e8f;
float scale = (0x0001) << 0;

void uint2uchar(unsigned int in, unsigned char* out){
  out[0] = (in & 0x00ff0000) >> 16;
  out[1] = (in & 0x0000ff00) >> 8;
  out[2] =  in & 0x000000ff;
}

boost::python::tuple render_mesh(boost::python::list proj_matrices,
                                 unsigned int im_height,
                                 unsigned int im_width,
                                 boost::python::numeric::array verts,
                                 boost::python::numeric::array tris,
                                 bool debug = false)
{
  // init rendering vars
  OSMesaContext ctx;
  boost::python::list color_ims;
  boost::python::list depth_ims;
  void *buffer;

  // parse input data
  int num_projections = boost::python::len(proj_matrices);
  long int verts_buflen;
  long int tris_buflen;
  void const *verts_raw_buffer;
  void const *tris_raw_buffer;
  bool verts_readbuf_success = !PyObject_AsReadBuffer(verts.ptr(), &verts_raw_buffer, &verts_buflen);
  bool tris_readbuf_success = !PyObject_AsReadBuffer(tris.ptr(), &tris_raw_buffer, &tris_buflen);
  const double* verts_buffer = reinterpret_cast<const double*>(verts_raw_buffer);
  const unsigned int* tris_buffer = reinterpret_cast<const unsigned int*>(tris_raw_buffer);

  unsigned int num_verts = verts_buflen / (3 * sizeof(double));
  unsigned int num_tris = tris_buflen / (3 * sizeof(unsigned int));
  if (debug) {
    std::cout << "Num vertices " << num_verts << std::endl;
    std::cout << "Num tris " << num_tris << std::endl;
  }

  // create an RGBA-mode context
  ctx = OSMesaCreateContextExt( OSMESA_RGBA, 16, 0, 0, NULL );
  if (!ctx) {
    printf("OSMesaCreateContext failed!\n");
  }
  
  // allocate the image buffer
  buffer = malloc( im_width * im_height * 4 * sizeof(GLubyte) );
  if (!buffer) {
    printf("Alloc image buffer failed!\n");
  }
  
  // bind the buffer to the context and make it current
  if (!OSMesaMakeCurrent( ctx, buffer, GL_UNSIGNED_BYTE, im_width, im_height )) {
    printf("OSMesaMakeCurrent failed!\n");
  }
  OSMesaPixelStore(OSMESA_Y_UP, 0);     
  
  // setup rendering
  glEnable(GL_DEPTH_TEST);   
  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  for (unsigned int k = 0; k < num_projections; k++) {
    // load next projection matrix
    boost::python::object proj_matrix_obj(proj_matrices[k]);
    long int proj_buflen;
    void const *proj_raw_buffer;    
    bool proj_readbuf_success = !PyObject_AsReadBuffer(proj_matrix_obj.ptr(),
                                                       &proj_raw_buffer,
                                                       &proj_buflen);    
    const double* projection = reinterpret_cast<const double*>(proj_raw_buffer);
    if (debug) {
      std::cout << "Proj Matrix " << k << std::endl;
      std::cout << projection[0] << " " << projection[1] << " " << projection[2] << " " << projection[3] << std::endl;
      std::cout << projection[4] << " " << projection[5] << " " << projection[6] << " " << projection[7] << std::endl;
      std::cout << projection[8] << " " << projection[9] << " " << projection[10] << " " << projection[11] << std::endl;
    }

    // create projection
    double final_matrix[16];
    double inv_width_scale  = 1.0 / (im_width * scale);
    double inv_height_scale = 1.0 / (im_height * scale);
    double inv_width_scale_1 = inv_width_scale - 1.0;
    double inv_height_scale_1_s = -(inv_height_scale - 1.0);
    double inv_width_scale_2 = inv_width_scale * 2.0;
    double inv_height_scale_2_s = -inv_height_scale * 2.0;
    double far_a_near = far + near;
    double far_s_near = far - near;
    double far_d_near = far_a_near / far_s_near;
    final_matrix[ 0] = projection[0+2*4] * inv_width_scale_1 + projection[0+0*4] * inv_width_scale_2;
    final_matrix[ 4] = projection[1+2*4] * inv_width_scale_1 + projection[1+0*4] * inv_width_scale_2;
    final_matrix[ 8] = projection[2+2*4] * inv_width_scale_1 + projection[2+0*4] * inv_width_scale_2; 
    final_matrix[ 12] = projection[3+2*4] * inv_width_scale_1 + projection[3+0*4] * inv_width_scale_2;

    final_matrix[ 1] = projection[0+2*4] * inv_height_scale_1_s + projection[0+1*4] * inv_height_scale_2_s;
    final_matrix[ 5] = projection[1+2*4] * inv_height_scale_1_s + projection[1+1*4] * inv_height_scale_2_s; 
    final_matrix[ 9] = projection[2+2*4] * inv_height_scale_1_s + projection[2+1*4] * inv_height_scale_2_s;
    final_matrix[13] = projection[3+2*4] * inv_height_scale_1_s + projection[3+1*4] * inv_height_scale_2_s;  

    final_matrix[ 2] = projection[0+2*4] * far_d_near;
    final_matrix[ 6] = projection[1+2*4] * far_d_near;
    final_matrix[10] = projection[2+2*4] * far_d_near;
    final_matrix[14] = projection[3+2*4] * far_d_near - (2*far*near)/far_s_near;

    final_matrix[ 3] = projection[0+2*4];
    final_matrix[ 7] = projection[1+2*4];
    final_matrix[11] = projection[2+2*4];
    final_matrix[15] = projection[3+2*4];

    // load projection and modelview matrices
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(final_matrix);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // render mesh
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, im_width, im_height);
    unsigned char colorBytes[3] = {255, 255, 255}; // all white
    for (unsigned int i = 0; i < num_tris; ++i) {
      glColor3ubv(colorBytes);
      glBegin(GL_POLYGON);
 
      unsigned int a = tris_buffer[3*i + 0];
      unsigned int b = tris_buffer[3*i + 1];
      unsigned int c = tris_buffer[3*i + 2];

      glVertex3dv(&verts_buffer[3 * a]);
      glVertex3dv(&verts_buffer[3 * b]);
      glVertex3dv(&verts_buffer[3 * c]);
      glEnd();
    }

    glFinish();

    // pull color buffer and flip y axis
    int i, j;
    GLint out_width, out_height, bytes_per_depth, color_type;
    GLboolean succeeded;
    unsigned char* p_color_buffer;
    succeeded = OSMesaGetColorBuffer(ctx, &out_width, &out_height, &color_type, (void**)&p_color_buffer);
    unsigned char* color_result = new unsigned char[3 * out_width * out_height];
    for (i = 0; i < out_width; i++) {
      for (j = 0; j < out_height; j++) {
        int di = i + j * out_width; // index in color buffer
        int ri = i + j * out_width; // index in rendered image
        color_result[3*ri+0] = p_color_buffer[4*di+0];
        color_result[3*ri+1] = p_color_buffer[4*di+1];
        color_result[3*ri+2] = p_color_buffer[4*di+2];
      }
    }

    // pull depth buffer and flip y axis
    unsigned short* p_depth_buffer;
    succeeded = OSMesaGetDepthBuffer(ctx, &out_width, &out_height, &bytes_per_depth, (void**)&p_depth_buffer);
    float* depth_result = new float[out_width * out_height];
    for(i = 0; i < out_width; i++){
      for(j = 0; j < out_height; j++){
        int di = i + j * out_width; // index in depth buffer
        int ri = i + (out_height-1-j)*out_width; // index in rendered image
        if (p_depth_buffer[di] == USHRT_MAX) {
          depth_result[ri] = 0.0f;
        }
        else {
          depth_result[ri] = near / (1.0f - ((float)p_depth_buffer[di] / USHRT_MAX));
        }
      }
    }

    // append ndarray color image to list
    boost::python::tuple color_shape = boost::python::make_tuple(im_height, im_width, 3);
    boost::numpy::dtype color_dt = boost::numpy::dtype::get_builtin<unsigned char>();
    boost::numpy::ndarray color_arr = boost::numpy::from_data(color_result, color_dt, color_shape,
                                                              boost::python::make_tuple(color_shape[0]*color_shape[2]*sizeof(unsigned char),
                                                                                        color_shape[2]*sizeof(unsigned char),
                                                                                        sizeof(unsigned char)),
                                                              boost::python::object());
    color_ims.append(color_arr);

    // append ndarray depth image to list
    boost::python::tuple depth_shape = boost::python::make_tuple(im_height, im_width);
    boost::numpy::dtype depth_dt = boost::numpy::dtype::get_builtin<float>();
    boost::numpy::ndarray depth_arr = boost::numpy::from_data(depth_result, depth_dt, depth_shape,
                                                              boost::python::make_tuple(depth_shape[0]*sizeof(float),
                                                                                        sizeof(float)),
                                                              boost::python::object());
    depth_ims.append(depth_arr);
  }
  
  // free the image buffer
  free( buffer );
  
  // destroy the context
  OSMesaDestroyContext( ctx );

  //return depth_ims;
  return boost::python::make_tuple(color_ims, depth_ims);
}

// Test function for multiplying an array by a scalar
boost::python::list mul_array(boost::python::numeric::array data, int x)
{ 
  // Access a built-in type (an array)
  boost::python::numeric::array a = data;
  long int bufLen;
  void const *buffer;
  bool isReadBuffer = !PyObject_AsReadBuffer(a.ptr(), &buffer, &bufLen);
  std::cout << "BUFLEN " << bufLen << std::endl;
  const double* test = reinterpret_cast<const double*>(buffer);
  int s = bufLen / sizeof(double);
  double* mult = new double[s];
  for (int i = 0; i < s; i++) {
    mult[i] = x * test[i];
  }

  const boost::python::tuple& shape = boost::python::extract<boost::python::tuple>(a.attr("shape"));
  std::cout << "Shape " << boost::python::extract<int>(shape[0]) << " " << boost::python::extract<int>(shape[1]) << std::endl;
  boost::numpy::dtype dt = boost::numpy::dtype::get_builtin<double>();
  boost::numpy::ndarray result = boost::numpy::from_data(mult, dt, shape,
                                                         boost::python::make_tuple(shape[0]*sizeof(double),
                                                                                   sizeof(double)),
                                                         boost::python::object());

  boost::python::list l;
  l.append(result);
  return l;
}
 
// Expose classes and methods to Python
BOOST_PYTHON_MODULE(meshrender) {
  Py_Initialize();
  boost::numpy::initialize();
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
 
  def("mul_array", &mul_array);
  def("render_mesh", &render_mesh);
}
