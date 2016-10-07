#include "ofApp.h"
#include "time.h"
#include <random>

using namespace ofxCv;
using namespace cv;

void ofApp::loadSettings() {
    // if you want to package the app by itself without an outer
    // folder, you can place the "data" folder inside the app in
    // the Resources folder (right click on the app, "show contents")
    //ofSetDataPathRoot("../Resources/data/");
    
    bGuiVisible = true;
    gui.setup();
    gui.setName("FaceOSC => Wekinator");
    gui.setPosition(0, 0);
    gui.add(bIncludePose.set("pose", true));
    gui.add(bIncludeGestures.set("gesture", true));
    gui.add(bIncludeAllVertices.set("raw", false));
    
    ofxXmlSettings xml;
    xml.loadFile("settings.xml");
    
    bool bUseCamera = true;
    
    xml.pushTag("source");
    if(xml.getNumTags("useCamera") > 0) {
        bUseCamera = xml.getValue("useCamera", 0);
    }
    xml.popTag();
    
    xml.pushTag("camera");
    if(xml.getNumTags("device") > 0) {
        cam.setDeviceID(xml.getValue("device", 0));
    }
    if(xml.getNumTags("framerate") > 0) {
        cam.setDesiredFrameRate(xml.getValue("framerate", 30));
    }
    camWidth = xml.getValue("width", 640);
    camHeight = xml.getValue("height", 480);
    cam.initGrabber(camWidth, camHeight);
    xml.popTag();
    
    xml.pushTag("movie");
    if(xml.getNumTags("filename") > 0) {
        string filename = ofToDataPath((string) xml.getValue("filename", ""));
        if(!movie.load(filename)) {
            ofLog(OF_LOG_ERROR, "Could not load movie \"%s\", reverting to camera input", filename.c_str());
            bUseCamera = true;
        }
        movie.play();
    }
    else {
        ofLog(OF_LOG_ERROR, "Movie filename tag not set in settings, reverting to camera input");
        bUseCamera = true;
    }
    if(xml.getNumTags("volume") > 0) {
        float movieVolume = ofClamp(xml.getValue("volume", 1.0), 0, 1.0);
        movie.setVolume(movieVolume);
    }
    if(xml.getNumTags("speed") > 0) {
        float movieSpeed = ofClamp(xml.getValue("speed", 1.0), -16, 16);
        movie.setSpeed(movieSpeed);
    }
    bPaused = false;
    movieWidth = movie.getWidth();
    movieHeight = movie.getHeight();
    xml.popTag();
    
    if(bUseCamera) {
        ofSetWindowShape(camWidth, camHeight);
        setVideoSource(true);
    }
    else {
        ofSetWindowShape(movieWidth, movieHeight);
        setVideoSource(false);
    }
    
    xml.pushTag("face");
    if(xml.getNumTags("rescale")) {
        tracker.setRescale(xml.getValue("rescale", 1.));
    }
    if(xml.getNumTags("iterations")) {
        tracker.setIterations(xml.getValue("iterations", 5));
    }
    if(xml.getNumTags("clamp")) {
        tracker.setClamp(xml.getValue("clamp", 3.));
    }
    if(xml.getNumTags("tolerance")) {
        tracker.setTolerance(xml.getValue("tolerance", .01));
    }
    if(xml.getNumTags("attempts")) {
        tracker.setAttempts(xml.getValue("attempts", 1));
    }
    bDrawMesh = true;
    if(xml.getNumTags("drawMesh")) {
        bDrawMesh = (bool) xml.getValue("drawMesh", 1);
    }
    if(xml.getNumTags("bIncludePose")) {
        bIncludePose = (bool) xml.getValue("bIncludePose", bIncludePose);
    }
    if(xml.getNumTags("bIncludeGestures")) {
        bIncludeGestures = (bool) xml.getValue("bIncludeGestures", bIncludeGestures);
    }
    if(xml.getNumTags("bIncludeAllVertices")) {
        bIncludeAllVertices = (bool) xml.getValue("bIncludeAllVertices", bIncludeAllVertices);
    }
    tracker.setup();
    xml.popTag();
    
    xml.pushTag("osc");
    host = xml.getValue("host", "localhost");
    port = xml.getValue("port", 6448);
    xml.popTag();
    
    osc.setup(host, port);
}

void ofApp::setup() {
    ofSetVerticalSync(true);
    loadSettings();
    
    
    //Learning
    trained = false;
    recording = false;
    
    //Sound:
    modulationFrequency = 4.1;
    modulationDepth = 0.2;
    centerFrequency = 512;
    resonance = 2;

    sampleRate 	= 44100; /* Sampling Rate */
    bufferSize	= 512; /* Buffer Size. you have to fill this buffer with sound using the for loop in the audioOut method */
    
    
    fft.setup(1024, 512, 256);
    oct.setup(44100, 1024, 10);
    
    ofxMaxiSettings::setup(sampleRate, 2, initialBufferSize);
    
    /*ofSetVerticalSync(true);
    ofEnableAlphaBlending();
    ofEnableSmoothing();
    
    ofSetSphereResolution(5);
     
    ofBackground(0,0,0); */
    
    ofSoundStreamSetup(2,2,this, sampleRate, bufferSize, 4); /* this has to happen at the end of setup - it switches on the DAC */
    
    
}

//--------------------------------------------------------------
void ofApp::exit(){
    ofSoundStreamStop();
    ofSoundStreamClose();
}


void ofApp::update() {
    if(bPaused)
        return;
    
    videoSource->update();
    if(videoSource->isFrameNew()) {
        tracker.update(toCv(*videoSource));
        sendFaceOsc(tracker);
        if (recording) {
            addTrainingExample(tracker);
        } else if (trained) {
            addRunExample(tracker);
        }
        rotationMatrix = tracker.getRotationMatrix();
    }
}

std::vector<double> ofApp::getInputs(ofxFaceTracker& tracker) {
    std::vector<double> input;
    
    //if(bIncludePose) {
    ofVec2f position = tracker.getPosition();
    input.push_back(position.x);
    input.push_back(position.y);
    
    //addMessage(tracker.getScale());
    input.push_back(tracker.getScale());
    
    ofVec3f orientation = tracker.getOrientation();
    input.push_back(orientation.x);
    input.push_back(orientation.y);
    input.push_back(orientation.z);
    // }
    
    //if (bIncludeGestures) {
    input.push_back(tracker.getGesture(ofxFaceTracker::MOUTH_WIDTH));
    input.push_back(tracker.getGesture(ofxFaceTracker::MOUTH_HEIGHT));
    input.push_back(tracker.getGesture(ofxFaceTracker::LEFT_EYEBROW_HEIGHT));
    input.push_back(tracker.getGesture(ofxFaceTracker::RIGHT_EYEBROW_HEIGHT));
    input.push_back(tracker.getGesture(ofxFaceTracker::LEFT_EYE_OPENNESS));
    input.push_back(tracker.getGesture(ofxFaceTracker::RIGHT_EYE_OPENNESS));
    input.push_back(tracker.getGesture(ofxFaceTracker::JAW_OPENNESS));
    input.push_back(tracker.getGesture(ofxFaceTracker::NOSTRIL_FLARE));
    //}
    
    /* if(bIncludeAllVertices){
     ofVec2f center = tracker.getPosition();
     vector<ofVec2f> imagePoints = tracker.getImagePoints();
     for(int i = 0; i < imagePoints.size(); i++){
     ofVec2f p = imagePoints.at(i);
     msg.addFloatArg((p.x - center.x)/tracker.getScale());
     msg.addFloatArg((p.y - center.y)/tracker.getScale());
     }
     } */

    return input;
}

void ofApp::addRunExample(ofxFaceTracker& tracker) {
    if(tracker.getFound()) {
        std::vector<double> input = getInputs(tracker);
        std::vector<double> output = myRegression.process(input);
        modulationFrequency = output[0];
        modulationDepth = output[1];
        centerFrequency = output[2];
        resonance = output[3];
     } else {
        // not found
    }
}




void ofApp::addTrainingExample(ofxFaceTracker& tracker) {
    if(tracker.getFound()) {
        //std::vector<double> input = getInputs(tracker);
        
        trainingExample tempExample;
       // tempExample.input = {f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14};
        tempExample.input = getInputs(tracker);
        tempExample.output = {modulationFrequency, modulationDepth, centerFrequency, resonance};
        trainingSet.push_back(tempExample);
        
    } else {
        // not found
    }
}


void ofApp::draw() {
    ofSetColor(255);
    videoSource->draw(0, 0);
    
    if(tracker.getFound()) {
        ofDrawBitmapString(ofToString((int) ofGetFrameRate()), ofGetWidth()-20, 40);
        
        if(bDrawMesh) {
            ofSetLineWidth(1);
            
            //tracker.draw();
            tracker.getImageMesh().drawWireframe();
            
            ofPushView();
            ofSetupScreenOrtho(sourceWidth, sourceHeight, -1000, 1000);
            ofVec2f pos = tracker.getPosition();
            ofTranslate(pos.x, pos.y);
            applyMatrix(rotationMatrix);
            ofScale(10,10,10);
            ofDrawAxis(tracker.getScale());
            ofPopView();
        }
    } else {
        ofDrawBitmapString("searching for face...", 240, 68);
    }
    
    if(bPaused) {
        ofSetColor(255, 0, 0);
        ofDrawBitmapString( "paused", 240, 84);
    }
    
    if(!bUseCamera) {
        ofSetColor(255, 0, 0);
        ofDrawBitmapString("speed "+ofToString(movie.getSpeed()), ofGetWidth()-100, 68);
    }
   /* ofSetColor(255, 255, 255);
    ofDrawBitmapString("Sending 14 inputs to Wekinator (/wek/inputs to port 6448):", 20, 20);
    ofDrawBitmapString("x- and y- position, scale, x y and z orientation, mouth width,",20, 32);
    ofDrawBitmapString("mouth height, left and right eyebrow height, left and right eye openness,",20, 44);
    ofDrawBitmapString("jaw openness, nostril flare",20, 56);
     */
    
    ofSetColor(160, 255, 240, 150);
    ofDrawBitmapString(":: RapidMix Regression example ::", 10,20);
    ofDrawBitmapString("Press space to randomize parameters. Hold 'R' to record mouse positions associated with current params.", 10,40);
    ofDrawBitmapString("Press 'T' to train models.", 10,60);
    ofDrawBitmapString("Click and drag to run model", 10,80);
    
    stringstream s;
    s << "mod frequency: " << modulationFrequency;
    ofDrawBitmapString(s.str(), 10,735);
    s.str("");
    s << "mod depth: " << modulationDepth;
    ofDrawBitmapString(s.str(), 10,750);
    s.str("");
    s << "center frequency: " << centerFrequency;
    ofDrawBitmapString(s.str(), 400,735);
    s.str("");
    s << "resonance: " << resonance;
    ofDrawBitmapString(s.str(), 400,750);
    s.str("");
    s << "trained: " << trained;
    ofDrawBitmapString(s.str(), 800,735);
    s.str("");
    s << "recording: " << recording;
    ofDrawBitmapString(s.str(), 800,750);
    s.str("");
    
    /* if(bGuiVisible) {
     gui.draw();
     } */
}

/*void ofApp::keyPressed(int key) {
    switch(key) {
        case 'r':
            tracker.reset();
            break;
        case 'm':
            bDrawMesh = !bDrawMesh;
            break;
        case 'p':
            bPaused = !bPaused;
            break;
        case 'g':
            bGuiVisible = !bGuiVisible;
            break;
        case OF_KEY_UP:
            movie.setSpeed(ofClamp(movie.getSpeed()+0.2, -16, 16));
            break;
        case OF_KEY_DOWN:
            movie.setSpeed(ofClamp(movie.getSpeed()-0.2, -16, 16));
            break;
    }
} */


//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    switch (key) {
        case 114:
            recording = true;
            break;
        case 116:
            trained = myRegression.train(trainingSet);
            break;
        case 32:
            std::random_device rd;
            std::default_random_engine generator(rd());
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            
            modulationFrequency = 4096 * distribution(generator);
            modulationDepth = distribution(generator);
            centerFrequency = 4096 * distribution(generator);
            resonance = 40 * distribution(generator);
            break;
    }
    
}

void ofApp::keyReleased(int key){
    switch (key) {
        case 114:
            recording = false;
            break;
    }
}


void ofApp::setVideoSource(bool useCamera) {
    
    bUseCamera = useCamera;
    
    if(bUseCamera) {
        videoSource = &cam;
        sourceWidth = camWidth;
        sourceHeight = camHeight;
    }
    else {
        videoSource = &movie;
        sourceWidth = movieWidth;
        sourceHeight = movieHeight;
    }
}


//--------------------------------------------------------------
void ofApp::audioOut(float * output, int bufferSize, int nChannels) {
    for (int i = 0; i < bufferSize; i++){
        double LFO01 = (( myLFO01.sinewave(modulationFrequency) + 1.0 )/ 2.0) * modulationDepth + (1 - modulationDepth);
        double oscOutput = myWave.pulse(83, LFO01);
        double myFilteredOutput = myFilter.lores(oscOutput, centerFrequency, resonance);
        
        if (fft.process(myFilteredOutput)) {
            oct.calculate(fft.magnitudes);
        }
        
        //play result
        mymix.stereo(myFilteredOutput, outputs, 0.5);
        output[i*nChannels    ] = outputs[0];
        output[i*nChannels + 1] = outputs[1];
        
    }
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){
   /* if (recording) {
        trainingExample tempExample;
        tempExample.input = {double(x), double(y)};
        tempExample.output = {modulationFrequency, modulationDepth, centerFrequency, resonance};
        trainingSet.push_back(tempExample);
    } */
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
   /* if (trained) {
        std::vector<double> input;
        input.push_back (double(x));
        input.push_back (double(y));
        std::vector<double> output = myRegression.process(input);
        modulationFrequency = output[0];
        modulationDepth = output[1];
        centerFrequency = output[2];
        resonance = output[3];
    } */
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    mouseDragged(x, y, button);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}
