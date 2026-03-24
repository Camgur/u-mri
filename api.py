from multiprocessing import freeze_support

from u_mri import train


def main() -> None:
	train.train("./01_dataset", epochs=500)


if __name__ == "__main__":
	freeze_support()
	main()
