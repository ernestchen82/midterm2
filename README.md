# midterm2
void publish_choice //publish the gesture events.if mbed's moves fit any one of the three classes(gesture_index < 3),then generate the publish message.
int PredictGesture  //Analyze the accelerator datas to obtain a prediction
int gestureClassify //setup tensorflow and generate gesture_index with PredictGesture function.
void capture        // RPC function starts select_thread,publish_thread,and calls gestureClassify run on select_thread.
void connect        //setup wifi and MQTT network and call publish_choice run on publish_thread every 1s
int main            //runs the RPC loop and start the mqtt_thread,then call connect into mqtt_queue that runs on mqtt_thread.
