import onnxruntime as rt
import numpy


def fun_execute_onnx(onnx_model_path, X_test):
    sess = rt.InferenceSession(onnx_model_path)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
    return pred_onx
