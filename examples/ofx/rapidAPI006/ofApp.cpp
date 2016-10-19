#include "ofApp.h"
#include "time.h"
#include <random>


//--------------------------------------------------------------
void ofApp::setup(){
    
    sender.setup(HOST, SENDPORT);
    receiver.setup(RECEIVEPORT);
    
    ofEnableAlphaBlending();
    ofSetupScreen();
    ofBackground(0, 0, 0);
    ofSetFrameRate(60);
    
    modulationFrequency = 4.1;
    modulationDepth = 0.2;
    centerFrequency = 512;
    resonance = 2;
    
    trained = false;
    recording = false;
    
    sampleRate 	= 44100; /* Sampling Rate */
    bufferSize	= 512; /* Buffer Size. you have to fill this buffer with sound using the for loop in the audioOut method */
    
    
    fft.setup(1024, 512, 256);
    oct.setup(44100, 1024, 10);
    
    int current = 0;
    ofxMaxiSettings::setup(sampleRate, 2, initialBufferSize);
    
    ofSetVerticalSync(true);
    ofEnableAlphaBlending();
    ofEnableSmoothing();
    
    ofSetSphereResolution(5);
    
    ofBackground(0,0,0);
    
    ofSoundStreamSetup(2,2,this, sampleRate, bufferSize, 4); /* this has to happen at the end of setup - it switches on the DAC */
    
    
}

//--------------------------------------------------------------
void ofApp::exit(){
    ofSoundStreamStop();
    ofSoundStreamClose();
}


//--------------------------------------------------------------
void ofApp::update(){
    
}

//--------------------------------------------------------------
void ofApp::draw(){
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
    
    ofNoFill();
    for(int i=0; i < oct.nAverages; i++) {
        ofSetColor(160,255, 240,
                   oct.averages[i] / 20.0 * 255.0);
        //		ofCircle(ofGetWidth() / 2, ofGetHeight()/2, i * 5);
        glPushMatrix();
        glTranslatef(ofGetWidth()/2,ofGetHeight()/2, 0);
        //glRotatef(0.01 * ofGetFrameNum() * speed * i, 0.01 * ofGetFrameNum() * speed * i,0.01 * ofGetFrameNum() * speed * i, 0);
        //		glutWireSphere(i * 5, 2 + (10 - (fabs(speed) * 10)), 2 + (fabs(speed) * 10));
        ofDrawSphere(0, 0, i * 5);
        glPopMatrix();
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

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    switch (key) {
        case 114:
            recording = false;
            break;
    }
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){
    if (recording) {
        trainingExample tempExample;
        tempExample.input = {double(x), double(y)};
        tempExample.output = {modulationFrequency, modulationDepth, centerFrequency, resonance};
        trainingSet.push_back(tempExample);
    }
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    if (trained) {
        std::vector<double> input;
        input.push_back (double(x));
        input.push_back (double(y));
        std::vector<double> output = myRegression.process(input);
        modulationFrequency = output[0];
        modulationDepth = output[1];
        centerFrequency = output[2];
        resonance = output[3];
    }
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    mouseDragged(x, y, button);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    
}