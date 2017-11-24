EMSCR=em++
BABEL=./node_modules/.bin/babel

# ----------------------------------------
#javascript for loading JSON
RAPID_JS = src/emscripten/rapidLibPost.js
RAPID_JS_BABEL = babel/$(RAPID_JS)
NODE_ENV_JS = src/emscripten/nodeEnv.js

#the .cpp files that are used
RL_KNN=src/knnClassification.cpp src/classification.cpp
RL_SVM=src/svmClassification.cpp dependencies/libsvm/libsvm.cpp
RL_NN=src/neuralNetwork.cpp src/regression.cpp
RL_MS=src/modelSet.cpp
RL_DTW=src/warpPath.cpp src/searchWindow.cpp src/dtw.cpp src/fastDTW.cpp src/seriesClassification.cpp
RL_STREAM=src/rapidStream.cpp dependencies/bayesfilter/src/BayesianFilter.cpp dependencies/bayesfilter/src/filter_utilities.cpp
SOURCE_RAPID = $(RL_MS) $(RL_NN) $(RL_KNN) $(RL_SVM) $(RL_DTW) $(RL_STREAM)

# destination .js file
OUTPUT_RAPID=rapidLib/RapidLib.js

# ----------------------------------------
# General flags
# https://kripken.github.io/emscripten-site/docs/tools_reference/emcc.html
# -s DEMANGLE_SUPPORT=1 
CFLAGS=-O3 -s DISABLE_EXCEPTION_CATCHING=0 -s ALLOW_MEMORY_GROWTH=1 -s ASSERTIONS=1 -s EXPORT_NAME="'RapidLib'" --memory-init-file 0

# ----------------------------------------
# Final paths
full: $(SOURCE_RAPID)
	$(BABEL) $(RAPID_JS) -d babel
	$(EMSCR) $(CFLAGS) -s MODULARIZE=1 --post-js $(RAPID_JS_BABEL) --bind -o $(OUTPUT_RAPID) $(SOURCE_RAPID)

test: $(SOURCE_RAPID)
	$(BABEL) $(RAPID_JS) -d babel
	$(EMSCR) $(CFLAGS) --pre-js $(NODE_ENV_JS) --post-js $(RAPID_JS_BABEL) --bind -o $(OUTPUT_RAPID) $(SOURCE_RAPID) --profiling
	mocha

node: $(SOURCE_RAPID)
	$(BABEL) $(RAPID_JS) -d babel
	$(EMSCR) $(CFLAGS) --pre-js $(NODE_ENV_JS) --post-js $(RAPID_JS_BABEL) --bind -o $(OUTPUT_RAPID) $(SOURCE_RAPID) --profiling
	mocha
	zip rapidLib/RapidLibNode $(OUTPUT_RAPID)	
	scp rapidLib/RapidLibNode.zip eaviuser@igor.doc.gold.ac.uk:/home/eavi/public_html/rapidmixapi.com/examples/RapidLib.js.zip

universal: $(SOURCE_RAPID)
	$(BABEL) $(RAPID_JS) -d babel --presets env --no-babelrc --module-id RapidLib
	$(EMSCR) $(CFLAGS) -s MODULARIZE=1 --pre-js $(NODE_ENV_JS) --post-js $(RAPID_JS_BABEL) --bind -o $(OUTPUT_RAPID) $(SOURCE_RAPID)

dev: full
	scp $(OUTPUT_RAPID) mzbys001@igor.doc.gold.ac.uk:/home/mzbys001/public_html/RapidMixLib.js

prod: full
	scp $(OUTPUT_RAPID) eaviuser@igor.doc.gold.ac.uk:/home/eavi/public_html/rapidmix/RapidLib.js

all: full