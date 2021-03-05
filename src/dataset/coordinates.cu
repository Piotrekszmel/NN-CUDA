#include "coordinates.cuh"

Coordinates::Coordinates(size_t batch_size, size_t num_batches) :
	batch_size(batch_size), num_batches(num_batches)
{
	std::ofstream f_0("src/dataset/coordinates_target0.txt");
	std::ofstream f_1("src/dataset/coordinates_target1.txt");

	for (int i = 0; i < num_batches; i++) {
		batches.push_back(Tensor(Shape(batch_size, 2)));
		targets.push_back(Tensor(Shape(batch_size, 1)));

		batches[i].allocMem();
		targets[i].allocMem();

		for (int k = 0; k < batch_size; k++) {
			batches[i][k] = static_cast<float>(rand()) / RAND_MAX - 0.5;
			batches[i][batches[i].getShape().x + k] = static_cast<float>(rand()) / RAND_MAX - 0.5;;
			if ( (batches[i][k] > 0 && batches[i][batches[i].getShape().x + k] > 0) ||
				 ((batches[i][k] < 0 && batches[i][batches[i].getShape().x + k] < 0)) ) {
				targets[i][k] = 1;
				f_1 << batches[i][k] << " " << batches[i][batches[i].getShape().x + k] << "\n";
			}
			else {
				targets[i][k] = 0;
				f_0 << batches[i][k] << " " << batches[i][batches[i].getShape().x + k] << "\n";
			}
		}

		batches[i].memCpy(HostToDevice);
		targets[i].memCpy(HostToDevice);
	}
	f_0.close();
	f_1.close();
}

int Coordinates::getNumBatches() {
	return num_batches;
}

std::vector<Tensor>& Coordinates::getBatches() {
	return batches;
}

std::vector<Tensor>& Coordinates::getTargets() {
	return targets;
}

void Coordinates::saveToFile(Tensor& batch,
							 Tensor& labels,
							 std::string path0,
							 std::string path1) {
	std::ofstream f_0(path0);
	std::ofstream f_1(path1);

	for (int k = 0; k < batch_size; k++) {
		if (labels[k] >= 0.5) {
			f_1 << batch[k] << " " << batch[batch.getShape().x + k] << "\n";
		}
		else {
			f_0 << batch[k] << " " << batch[batch.getShape().x + k] << "\n";
		}
	}
}