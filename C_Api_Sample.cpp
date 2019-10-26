// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

//*****************************************************************************
// helper function to check for status
void CheckStatus(OrtStatus* status)
{
    if (status != NULL) {
      const char* msg = g_ort->GetErrorMessage(status);
      fprintf(stderr, "%s\n", msg);
      g_ort->ReleaseStatus(status);
      exit(1);
    }
}

int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  OrtEnv* env;
  CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  // initialize session options if needed
  OrtSessionOptions* session_options;
  CheckStatus(g_ort->CreateSessionOptions(&session_options));
  g_ort->SetIntraOpNumThreads(session_options, 1);

  // Sets graph optimization level
  g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

  // Optionally add more execution providers via session_options
  // E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
  // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

  //*************************************************************************
  // create session and load model into memory
  // using squeezenet version 1.3
  // URL = https://github.com/onnx/models/tree/master/squeezenet
  OrtSession* session;
#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
#else
  const char* model_path = "squeezenet.onnx";
#endif

  printf("Using Onnxruntime C API\n");
  CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session));

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  size_t num_input_nodes;
  OrtStatus* status;
  OrtAllocator* allocator;
  CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  // print number of model input nodes
  status = g_ort->SessionGetInputCount(session, &num_input_nodes);
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name;
    status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
    printf("Input %zu : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    OrtTypeInfo* typeinfo;
    status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info;
	CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
	CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
    printf("Input %zu : type=%d\n", i, type);

    // print input shapes/dims
    size_t num_dims;
	CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
    printf("Input %zu : num_dims=%zu\n", i, num_dims);
    input_node_dims.resize(num_dims);
	g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
    for (size_t j = 0; j < num_dims; j++)
      printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);

	g_ort->ReleaseTypeInfo(typeinfo);
  }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"softmaxout_1"};

  // initialize input data with values in [0.0, 1.0]
  for (size_t i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);


  //++++ 1.START NEW CODE: generate input_tensor_values2 for input_tensor2 
  std::vector<float> input_tensor_values2(input_tensor_size);
  std::vector<const char*> output_node_names2 = {"softmaxout_1"};
  // initialize input data with values in [0.0, 1.0]
  for (size_t i = 0; i < input_tensor_size; i++)
    input_tensor_values2[i] = 0.5;
  //++++ 1.END NEW CODE

  
  // create input tensor object from data values
  OrtMemoryInfo* memory_info;
  CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  OrtValue* input_tensor = NULL;
  CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

  int is_tensor;
  CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);


  //++++ 2.START NEW CODE: create input_tensor2 from generated input_tensor_values2
  OrtValue* input_tensor2 = NULL;
  CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values2.data(), input_tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor2));
  //++++ 2.END NEW CODE

  
  g_ort->ReleaseMemoryInfo(memory_info);

  // score model & input tensor, get back output tensor
  OrtValue* output_tensor = NULL;
  CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor));
  CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
  assert(is_tensor);

  // Get pointer to output tensor float values
  float* floatarr;
  CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  printf("For output_tensor %p\n", output_tensor);
  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score %p for class [%d] =  %f\n", floatarr, i, floatarr[i]);

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317

  //For output_tensor 0x55856f70cc60
  //Score 0x7f71eed74040 for class [0] =  0.000045
  //Score 0x7f71eed74040 for class [1] =  0.003846
  //Score 0x7f71eed74040 for class [2] =  0.000125
  //Score 0x7f71eed74040 for class [3] =  0.001180
  //Score 0x7f71eed74040 for class [4] =  0.001317


  //++++ 3.START NEW CODE: score input_tensor2, get output_tensor2, and print results  
  // score model & input tensor, get back output tensor
  OrtValue* output_tensor2 = NULL;
  CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor2, 1, output_node_names.data(), 1, &output_tensor2));
  CheckStatus(g_ort->IsTensor(output_tensor2, &is_tensor));
  assert(is_tensor);

  // Get pointer to output tensor float values
  float* floatarr2;
  CheckStatus(g_ort->GetTensorMutableData(output_tensor2, (void**)&floatarr2));
  //assert(abs(floatarr2[0] - 0.000045) < 1e-6);

  printf("For output_tensor2 %p\n", output_tensor2);
  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score %p for class [%d] =  %f\n", floatarr2, i, floatarr2[i]);

  //For output_tensor2 0x55856f716080
  //Score 0x7f71eed73040 for class [0] =  0.000048
  //Score 0x7f71eed73040 for class [1] =  0.003449
  //Score 0x7f71eed73040 for class [2] =  0.000113
  //Score 0x7f71eed73040 for class [3] =  0.001138
  //Score 0x7f71eed73040 for class [4] =  0.001164
  //++++ 3.END NEW CODE


  //++++ 4.START NEW CODE: score for input_tensors in an array, get output_tensors in an array, and print results  
  std::vector<OrtValue*> input_tensors(2);
  input_tensors[0] = input_tensor;
  input_tensors[1] = input_tensor2;
  
  std::vector<OrtValue*> output_tensors(2);
  output_tensors[0] = NULL;
  output_tensors[1] = NULL;

  std::vector<const char*> input_names(2);
  input_names[0] = input_node_names[0];
  input_names[1] = input_node_names[0]; // two inputs have the same names
  
  std::vector<const char*> output_names(2);
  output_names[0] = output_node_names[0];
  output_names[1] = output_node_names[0]; // two outputs have the same names

  // ORT Run
  CheckStatus(g_ort->Run(session, NULL, input_names.data(), (const OrtValue* const*)input_tensors.data(), 2, output_names.data(), 2, output_tensors.data()));

  // print results, use variables floatarr3 and floatarr4 to illustrate, could have looped
  // use floatarr3 to get tensor mutable data from output_tensors[0] 
  float* floatarr3;
  printf("Score for output_tensors[0] %p\n", output_tensors[0]);
  CheckStatus(g_ort->GetTensorMutableData(output_tensors[0], (void**)&floatarr3));
  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
      printf("Score %p for class [%d] =  %f\n", floatarr3, i, floatarr3[i]);

  // use floatarr4 to get tensor mutable data from output_tensors[1] 
  float* floatarr4;
  printf("Score for output_tensors[1] %p\n", output_tensors[1]);
  CheckStatus(g_ort->GetTensorMutableData(output_tensors[1], (void**)&floatarr4));
  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
      printf("Score %p for class [%d] =  %f\n", floatarr4, i, floatarr4[i]);

  //Score for output_tensors[0] 0x55856f721d50
  //Score 0x7f71eed76040 for class [0] =  0.000048
  //Score 0x7f71eed76040 for class [1] =  0.003449
  //Score 0x7f71eed76040 for class [2] =  0.000113
  //Score 0x7f71eed76040 for class [3] =  0.001138
  //Score 0x7f71eed76040 for class [4] =  0.001164
  //Score for output_tensors[1] 0x55856f84c6b0
  //Score 0x7f71eed76040 for class [0] =  0.000048
  //Score 0x7f71eed76040 for class [1] =  0.003449
  //Score 0x7f71eed76040 for class [2] =  0.000113
  //Score 0x7f71eed76040 for class [3] =  0.001138
  //Score 0x7f71eed76040 for class [4] =  0.001164

  // PROBLEM FOUND!!!!
  // output_tensors[0] and output_tensors[1] printed the same results
  // output_tensors[0] looks overwrittern by output_tensors[1]
  // the floatarr3 and floatarr4 happended to share the same pointer address 0x7f71eed76040,
  // though they are from separate GetTensorMutableData() calls
  // where could it be wrong in the new codes?
  
  //++++ 4.END NEW CODE

  
  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);
  
  //++++ 5.START NEW CODE: clean up output_tensor2
  g_ort->ReleaseValue(output_tensor2);
  g_ort->ReleaseValue(input_tensor2);
  //++++ 5.END NEW CODE
  
  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);
  printf("Done!\n");
  return 0;
}
