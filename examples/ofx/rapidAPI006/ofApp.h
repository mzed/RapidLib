#pragma once

#include "ofMain.h"
#include "ofxMaxim.h"
#include "ofxMaxim.h"
#include "maxiGrains.h"
#include "ofxOsc.h"
#include "regression.h"
#include <sys/time.h>

#define HOST "localhost"
#define RECEIVEPORT 12000
#define SENDPORT 6448

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void exit();
    void update();
    void draw();
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y);
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    //
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
    
    regression myRegression;
    std::vector<trainingExample> trainingSet;
    bool recording;
    bool trained;
    
    //osc
    ofxOscSender sender;
    ofxOscReceiver receiver;

    bool isTraining;
    
};
