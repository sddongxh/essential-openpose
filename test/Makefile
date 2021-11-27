CC = g++
CPPFLAGS = -g -Wall -O3  -MMD  -std=c++17

TARGET = test.exe

#extra libs
LIBS += -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_highgui 
LIBS += -lcaffe-nv -lglog -lprotobuf
LIBS += ../core/libopenpose_core.so ../net/libopenpose_net.so #../utilities/libopenpose_utilities
# sources & includes
SRC_PATHS =  .
INC_PATHS +=  ../include /usr/local/cuda/include #../caffe/include ../caffe/3rdparty


SRCS += $(foreach path, $(SRC_PATHS), $(wildcard $(path)/*.cpp))
INCS += $(foreach path, $(INC_PATHS), -I$(path))

OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

CPPFLAGS += $(INCS)

all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CPPFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp 
	@$(CC) $(CPPFLAGS) -c $< -o $@

-include $(DEPS)

clean: 
	rm -rf $(OBJS)  $(TARGET)  $(DEPS)
