# Simple Makefile to handle release/debug modes
# As well as other CMake command line args
# (which are hard to type and remember)

all: release

release:
	mkdir -p build_release
	cd build_release && cmake -DCMAKE_BUILD_TYPE=RELEASE ..
	$(MAKE) -C ./build_release

debug:
	mkdir -p build_debug
	cd build_debug && cmake -DCMAKE_BUILD_TYPE=DEBUG ..
	$(MAKE) -C ./build_debug

test: release
	./build_release/bin/unittests

test_release: test

test_debug: debug
	./build_debug/bin/unittests

metal:
	mkdir -p build_metal
	cd build_metal && cmake -DCMAKE_BUILD_TYPE=RELEASE \
		-DESSENTIALS_METAL_BACKEND=ON \
		-DESSENTIALS_AMD_BACKEND=OFF \
		-DESSENTIALS_NVIDIA_BACKEND=OFF \
		-DESSENTIALS_BUILD_EXAMPLES=OFF ..
	$(MAKE) -C ./build_metal

metal-debug:
	mkdir -p build_metal_debug
	cd build_metal_debug && cmake -DCMAKE_BUILD_TYPE=DEBUG \
		-DESSENTIALS_METAL_BACKEND=ON \
		-DESSENTIALS_AMD_BACKEND=OFF \
		-DESSENTIALS_NVIDIA_BACKEND=OFF \
		-DESSENTIALS_BUILD_EXAMPLES=OFF ..
	$(MAKE) -C ./build_metal_debug

test_metal: metal
	cd build_metal && ctest --output-on-failure

clean:
	rm -rf build_debug
	rm -rf build_release
	rm -rf build_metal
	rm -rf build_metal_debug
	rm -rf externals
