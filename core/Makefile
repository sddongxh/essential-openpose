CC = g++
CPPFLAGS = -g -Wall -O3 -MMD -std=c++17 -mavx2 -mfma -fopenmp 
CPPFLAGS += -fPIC -shared 

TARGET = libopenpose_core.so

#extra libs
LIBS += -lopencv_core
LIBS += /home/xihua/code/essential-openpose/utilities/libopenpose_utilities.so
# sources & includes
SRC_PATHS = ./src

INC_PATHS =  ../include 

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