# Build Command
cmake -DESSENTIALS_NVIDIA_BACKEND=ON -DESSENTIALS_AMD_BACKEND=OFF  -DCMAKE_CUDA_ARCHITECTURES=121  -DESSENTIALS_BUILD_BENCHMARKS=OFF 

cd /home/ss20458/svishnus-code/gunrock/build
make bc -j$(nproc)


# BFS
./bin/bfs -m ../datasets/roadNet-CA/roadNet-CA.mtx --src 0 --advance_load_balance merge_path --validate
./bin/bfs -m ../datasets/soc-LiveJournal1/soc-LiveJournal1.mtx --src 0 --advance_load_balance merge_path

# SSSP
./bin/sssp -m ../datasets/roadNet-CA/roadNet-CA.mtx --validate
./bin/sssp -m ../datasets/road_usa/road_usa.mtx --validate

# PageRank
./bin/pr -m ../datasets/soc-LiveJournal1/soc-LiveJournal1.mtx
./bin/pr -m ../datasets/hollywood-2009/hollywood-2009.mtx

# Triangle Counting
./bin/tc -m ../datasets/roadNet-CA/roadNet-CA.mtx --validate

# Betweenness Centrality
./bin/bc -m ../datasets/roadNet-CA/roadNet-CA.mtx

# Graph Coloring
./bin/color -m ../datasets/soc-LiveJournal1/soc-LiveJournal1.mtx

# K-Core
./bin/kcore -m ../datasets/soc-LiveJournal1/soc-LiveJournal1.mtx

# MST
./bin/mst -m ../datasets/roadNet-CA/roadNet-CA.mtx --validate

# HITS
./bin/hits -m ../datasets/hollywood-2009/hollywood-2009.mtx

# SpMV
./bin/spmv -m ../datasets/soc-orkut/soc-orkut.mtx
