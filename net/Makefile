CC = g++
CPPFLAGS = -g -Wall -O3 -MMD -std=c++17 -mavx2 -mfma -fopenmp 
CPPFLAGS += -fPIC -shared 

# CPPFLAGS  += -DNV_CAFFE


TARGET = libopenpose_net.so

#extra libs
LIBS += -lopencv_core -lcaffe-nv
LIBS += /home/xihua/code/essential-openpose/core/libopenpose_core.so 
# sources & includes
SRC_PATHS = ./src

INC_PATHS =  ../include  /usr/local/cuda/include

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
	rm -rf $(OBJS)  $(TARGET) 

clean-app:
	rm -rf $(TARGET) src/*.o src/*.d