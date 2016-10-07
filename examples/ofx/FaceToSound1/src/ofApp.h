#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "ofxCv.h"
#include "FaceOsc.h"
#include "ofxXmlSettings.h"
#include "regression.h"
#include "maxiGrains.h"
#include "ofxMaxim.h"
#include <sys/time.h>

class ofApp : public ofBaseApp, public FaceOsc {
public:
    void loadSettings();
    
    void setup();
    void update();
    void exit();
    void draw();
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y);
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    
    std::vector<double> getInputs(ofxFaceTracker& tracker);
    
    void setVideoSource(bool useCamera);
    
    bool bUseCamera, bPaused;
    
    int camWidth, camHeight;
    int movieWidth, movieHeight;
    int sourceWidth, sourceHeight;
    
    ofVideoGrabber cam;
    ofVideoPlayer movie;
    ofBaseVideoDraws *videoSource;
    
    ofxFaceTracker tracker;
    ofMatrix4x4 rotationMatrix;
    
    ofxPanel gui;
    
    bool bDrawMesh;
    bool bGuiVisible;
    
    //For training:
    bool isTraining;
    regression myRegression;
    std::vector<trainingExample> trainingSet;
    bool recording;
    bool trained;
    void addTrainingExample(ofxFaceTracker& tracker);
    void addRunExample(ofxFaceTracker& tracker);
    
    //For audio:
    double modulationFrequency, modulationDepth;
    double centerFrequency, resonance;
    
    void audioOut(float * output, int bufferSize, int nChannels);
    
    int	bufferSize;
    
    int	initialBufferSize; /* buffer size */
    int	sampleRate;
    ofxMaxiFFT fft;
    ofxMaxiFFTOctaveAnalyzer oct;
    int current;
    double pos;
    
    double oscOutput, outputs[2];
    ofxMaxiOsc myWave, myLFO01;
    ofxMaxiFilter myFilter;
    maxiMix mymix;
    
};