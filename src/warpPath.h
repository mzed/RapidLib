//
//  warpPath.h
//  RapidAPI
//
//  Created by mzed on 13/09/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#ifndef warpPath_h
#define warpPath_h

class warpPath {
public:
    std::vector<int> xIndices;
    std::vector<int> yIndices;
    
    void add(int x, int y) {
        xIndices.insert(xIndices.begin(), x);
        yIndices.insert(yIndices.begin(), y);
    }

};
#endif /* warpPath_h */
