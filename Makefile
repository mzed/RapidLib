# change '../../em++' to path to em++ in emscripten folder
EMSCR=/Users/mzed/emsdk_portable/emscripten/1.35.0/em++ 

# ----------------------------------------
#javascript for loading JSON
RAPID_JS = src/rapidMix.js

#the .cpp files that are used
WEKI_KNN=src/knnClassification.cpp src/classification.cpp
WEKI_NN=src/neuralNetwork.cpp src/regression.cpp
WEKI_MS=src/modelSet.cpp 
SOURCE_WEKI = $(WEKI_MS) $(WEKI_NN) $(WEKI_KNN)

# destination .js file
OUTPUT_WEKI=wekiLib/RapidMixLib.js

# ----------------------------------------
# General flags
# https://kripken.github.io/emscripten-site/docs/tools_reference/emcc.html
# -s DEMANGLE_SUPPORT=1 
CFLAGS=-O3 -s DISABLE_EXCEPTION_CATCHING=0 -s ALLOW_MEMORY_GROWTH=1 -s ASSERTIONS=1 -s EXPORT_NAME="'RapidLib'" --memory-init-file 0 --profiling

# ----------------------------------------
# Final paths
full: $(SOURCE_WEKI)
	$(EMSCR) $(CFLAGS) -s MODULARIZE=1 --post-js $(RAPID_JS) --bind -o $(OUTPUT_WEKI) $(SOURCE_WEKI)

test: $(SOURCE_WEKI)
	$(EMSCR) $(CFLAGS) --post-js $(RAPID_JS) --bind -o $(OUTPUT_WEKI) $(SOURCE_WEKI)
	mocha
push: full
	scp wekiLib/RapidMixLib.js mzbys001@igor.doc.gold.ac.uk:/home/mzbys001/public_html/

all: full