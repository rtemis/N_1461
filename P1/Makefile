# McCulloch-Pitts
ayuda_mp:
	@echo "----------------------------------------------------"
	@echo "HELP Neural Network - McCulloch-Pitts"
	@echo "	-- No arguments available"
	@echo "----------------------------------------------------"

compila_mp:
	@echo "----------------------------------------------------"
	@echo "Compiling Neural Network - McCulloch-Pitts"
	@echo "--Nothing to do"
	@echo "----------------------------------------------------"

ejecuta_mp:
	@echo "----------------------------------------------------"
	@echo "Executing Neural Network - McCulloch-Pitts"
	@echo "----------------------------------------------------"
	python3 mcculloch.py 

# Compile all
compilar:
	@echo "----------------------------------------------------"
	@echo "Compiling Neural Network - Perceptron"
	@echo "--Nothing to do"
	@echo "Compiling Neural Network - Adaline"
	@echo "--Nothing to do"
	@echo "----------------------------------------------------"

# Perceptron
ayuda_per:
	@echo "----------------------------------------------------"
	@echo "HELP Neural Network - Perceptron"
	@echo "	data=<value>		Value 1, 2, or 3, mode in which file is read	default=1"
	@echo "	alpha=<value>		Value between 0.0 and 1.0	default=1"
	@echo "	umbral=<value> 		Value between 0.0 and 1.0	default=0.2"
	@echo "	prop=<value>		Value between 0.0 and 1.0	default=0.7"
	@echo "	tolerance=<value> 	Value between 0.0 and 1.0	default=0.0001"
	@echo "	patience=<value>	Integer value max iterations	default=200"
	@echo "	train=<file> 		Training file name		default=\"data/problema_real1.txt\""
	@echo "	test=<file> 		Testing file name		default=\"data/problema_real1.txt\""
	@echo "----------------------------------------------------"

ejecuta_per:
	@echo "----------------------------------------------------"
	@echo "Executing Neural Network - Perceptron"
	@echo "----------------------------------------------------"
	@python3 perceptron.py --alpha $(alpha) --data $(data) --umbral $(umbral) --prop $(prop) --tolerance $(tolerance) --patience $(patience) --train $(train) --test $(test) 

# Adaline
ayuda_ada:
	@echo "----------------------------------------------------"
	@echo "HELP Neural Network - Adaline"
	@echo "	data=<value>		Value 1, 2, or 3, mode in which file is read	default=1"
	@echo "	alpha=<value>		Value between 0.0 and 1.0	default=1"
	@echo "	umbral=<value> 		Value between 0.0 and 1.0	default=0.2"
	@echo "	prop=<value>		Value between 0.0 and 1.0	default=0.7"
	@echo "	tolerance=<value> 	Value between 0.0 and 1.0	default=0.0001"
	@echo "	patience=<value>	Integer value max iterations	default=200"
	@echo "	train=<file> 		Training file name		default=\"data/problema_real1.txt\""
	@echo "	test=<file> 		Testing file name		default=\"data/problema_real1.txt\""
	@echo "----------------------------------------------------"

ejecuta_ada:
	@echo "----------------------------------------------------"
	@echo "Executing Neural Network - Adaline"
	@echo "----------------------------------------------------"
	@python3 adaline.py --alpha $(alpha) --data $(data) --umbral $(umbral) --prop $(prop) --tolerance $(tolerance) --patience $(patience) --train $(train) --test $(test) 

