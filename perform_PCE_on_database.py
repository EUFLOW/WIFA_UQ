import openturns as ot
from operator import itemgetter
import numpy as np

        
def construct_PCE_ot(training_input, training_output, \
                     marginals, copula, degree, LARS=True):

    ##########################INPUTS##########################
    ##########################################################
    Nt = len(training_input)
    if(len(training_input.shape) > 1):
        Nvar = training_input.shape[1]
    else:
        Nvar = 1

    #Define Sample
    outputSample = ot.Sample(Nt,1)
    for i in range(Nt):
        outputSample[i,0] = training_output[i]

    #Define Collection and PDFs
    polyColl = ot.PolynomialFamilyCollection(Nvar)    
    collection = ot.DistributionCollection(Nvar)
    marginal = {}
    UncorrelatedInputSample = ot.Sample(Nt,Nvar)

    if(Nvar>1):
        for i in range(Nvar):
            varSample = ot.Sample(Nt,1)
            for j in range(Nt):
                varSample[j,0] = training_input[j,i]
                UncorrelatedInputSample[j,i] = training_input[j,i]
            minValue = varSample.getMin()[0]
            maxValue = varSample.getMax()[0]
            if(marginals[i]=="gaussian" or marginals[i]=="normal"):
                marginal[i] = ot.NormalFactory().build(varSample)
            elif(marginals[i]=="uniform"):
                marginal[i] = ot.Uniform(minValue-minValue/100.,maxValue+maxValue/100.)
            elif(marginals[i]=="kernel"):
                marginal[i] = ot.KernelSmoothing().build(varSample)
            else:
                print("WARNING: couldn't find distribution '"+str(marginals[i])+"', applied kernel smoothing instead")
                marginal[i] = ot.KernelSmoothing().build(varSample)

            collection[i] = ot.Distribution(marginal[i])
    else:
        varSample = ot.Sample(Nt,1)
        for j in range(Nt):
            varSample[j,0] = training_input[j]
            UncorrelatedInputSample[j,0] = training_input[j]
        minValue = varSample.getMin()[0]
        maxValue = varSample.getMax()[0]
        if(marginals[i]=="gaussian" or marginals[i]=="normal"):  
            marginal[0] = ot.NormalFactory().build(varSample)
        elif(marginals[i]=="uniform"):
            marginal[0] = ot.Uniform(minValue-minValue/100.,maxValue+maxValue/100.)
        elif(marginals[i]=="kernel"):
            marginal[0] = ot.KernelSmoothing().build(varSample)
        else:
            print("WARNING: couldn't find distribution '"+str(marginals[i])+"', applied kernel smoothing instead")
            marginal[0] = ot.KernelSmoothing().build(varSample)
        collection[0] = ot.Distribution(marginal[0])

    if(copula=="independent"):
        copula = ot.IndependentCopula(Nvar)
    elif(copula=="gaussian" or copula=="normal"):
        inputSample = ot.Sample(training_input)
        copula = ot.NormalCopulaFactory().build(inputSample)
    else:
        print("WARNING: couldn't find copula '"+str(copula)+"', applied independent copula instead")
        copula = ot.IndependentCopula(Nvar)

    #UncorrelatedInputDistribution = ot.ComposedDistribution(collection,ot.Copula(copula))
    UncorrelatedInputDistribution = ot.ComposedDistribution(collection,copula)

    #Calcul des polynomes du chaos
    for v in range(0,Nvar):
        marginalv=UncorrelatedInputDistribution.getMarginal(v)            
        if(marginals[i]=="kernel"):
            #Works with arbitrary PDF
            basisAlgorithm = ot.AdaptiveStieltjesAlgorithm(marginalv)
            polyColl[v] = ot.StandardDistributionPolynomialFactory(basisAlgorithm)
        else:
            #Works with standard PDF: gaussian, uniform, ..
            polyColl[v] = ot.StandardDistributionPolynomialFactory(marginalv)

    # Definition de la numerotation des coefficients des polynomes du chaos
    enumerateFunction = ot.LinearEnumerateFunction(Nvar)
    #enumerateFunction = HyperbolicAnisotropicEnumerateFunction(Nvar,0.4)
    # Creation de la base des polynomes multivaries en fonction de la numerotation
    #                     et des bases desiree
    multivariateBasis = ot.OrthogonalProductPolynomialFactory(polyColl,enumerateFunction)
    # Number of PC terms
    P = enumerateFunction.getStrataCumulatedCardinal(degree)
    #Troncature
    adaptativeStrategy = ot.FixedStrategy(multivariateBasis,P)


    if(LARS):
        #Evaluation Strategy : LARS
        basisSequenceFactory = ot.LARS()
        fittingAlgorithm = ot.CorrectedLeaveOneOut()
        approximationAlgorithm = ot.LeastSquaresMetaModelSelectionFactory(basisSequenceFactory, fittingAlgorithm)

        #Approximation method for PCE coefficients
        projectionStrategy = ot.LeastSquaresStrategy(UncorrelatedInputSample, outputSample, approximationAlgorithm)
        #
        algo = ot.FunctionalChaosAlgorithm(UncorrelatedInputSample, outputSample, UncorrelatedInputDistribution, adaptativeStrategy, projectionStrategy)
    else:
        #
        wei_exp = ot.MonteCarloExperiment(UncorrelatedInputDistribution,UncorrelatedInputSample.getSize())
        X_UncorrelatedInputSample, weights = wei_exp.generateWithWeights()
        projectionStrategy = ot.LeastSquaresStrategy()
        algo = ot.FunctionalChaosAlgorithm(X_UncorrelatedInputSample, weights, outputSample, UncorrelatedInputDistribution, adaptativeStrategy, projectionStrategy)

    algo.run()
    polynomialChaosResult = algo.getResult()
    metamodel = polynomialChaosResult.getMetaModel()
    enumerateFunction = enumerateFunction

    return polynomialChaosResult
        
def build_PCE_for_given_time(UQ_mapping_results, model_param_varnames, output_variable_name, \
                             it, PCE_deg, marginals, copula):
    
    #########Set the data UQ_mapping_results###############
    stochastic_nvar = len(model_param_varnames)
    sample_size=len(UQ_mapping_results.sample)
    #
    input_variable_array = np.zeros((sample_size,stochastic_nvar))
    for v in range(stochastic_nvar):
        input_variable_array[:,v] = UQ_mapping_results.isel().get(model_param_varnames[v])
        #
        output_variable_array = np.zeros((sample_size))
        output_variable_array[:] = UQ_mapping_results.isel(time=it).get(output_variable_name)[:,0]
        
    #########Fit PCE with chosen parameters############
    polynomialChaosResult = construct_PCE_ot(input_variable_array, \
                                             output_variable_array, marginals, \
                                             copula, PCE_deg, LARS=True) #LARS=True if sparse
    chaosSI = ot.FunctionalChaosSobolIndices(polynomialChaosResult)

    #########Get Sobol indices############
    
    Sobol_indices_1stOrder = [chaosSI.getSobolIndex(i) for i in range(stochastic_nvar)]
    Sobol_indices_Total =  [chaosSI.getSobolTotalIndex(i) for i in range(stochastic_nvar)]
    PCE_metamodel = polynomialChaosResult.getMetaModel()
    
    return Sobol_indices_1stOrder, Sobol_indices_Total, PCE_metamodel 

def build_PCE_for_all_times(UQ_mapping_results, model_param_varnames, output_variable_name, \
                             PCE_deg, marginals, copula):
    #
    physical_size = len(UQ_mapping_results.time)
    allTimes_Sobol_indices_1stOrder = np.zeros((physical_size, len(model_param_varnames)))
    allTimes_Sobol_indices_Total = np.zeros((physical_size, len(model_param_varnames)))
    allTimes_PCE_metamodels = []
    #
    for it in range(physical_size):
        Sobol_indices_1stOrder, Sobol_indices_Total, PCE_metamodel  =  build_PCE_for_given_time(UQ_mapping_results, model_param_varnames, \
                                                                                                output_variable_name,it,PCE_deg, marginals, copula)
        allTimes_Sobol_indices_1stOrder[it,:] = Sobol_indices_1stOrder
        allTimes_Sobol_indices_Total[it,:] = Sobol_indices_Total
        allTimes_PCE_metamodels.append(PCE_metamodel)

    return allTimes_Sobol_indices_1stOrder, allTimes_Sobol_indices_Total, allTimes_PCE_metamodels
def build_POD_PCE(UQ_mapping_results, model_param_varnames, output_variable_name, \
                             PCE_deg, marginals, copula,target_POD_EVR=0.99):
    ##### Perform POD to get variations and eignevalues
    #########Set the data from previous###############
    complete_output_variable_array = UQ_mapping_results.isel().get(output_variable_name)[:,:,0]    
    #######SKLEARN PCA##########
    from sklearn.decomposition import PCA
    pca = PCA(svd_solver='full')
    pca.fit(complete_output_variable_array)
    singular_value_matrix = np.diag(pca.singular_values_)
    EVR = pca.explained_variance_ratio_
    cumulative_EVR = np.array([np.sum(EVR[:i]) for i in range(len(EVR))])
    #print(cumulative_EVR[:10])
    selected_mode_number = np.where(cumulative_EVR>target_POD_EVR)[0][0] + 1
    #
    stochastic_coefficients = pca.transform(complete_output_variable_array)[:,:selected_mode_number]
    POD_basis = (pca.components_).transpose()[:,:selected_mode_number]
    singular_values = pca.singular_values_[:selected_mode_number]
    #print(stochastic_coefficients.shape)
    #print(POD_basis.shape)
    #print(singular_values.shape)
    #temporal_basis = (np.dot(np.linalg.inv(singular_value_matrix),np.dot(POD_basis.transpose(),complete_output_variable_array.transpose()))).transpose()

    #########Set the data UQ_mapping_results###############
    stochastic_nvar = len(model_param_varnames)
    sample_size=len(UQ_mapping_results.sample)
    #
    input_variable_array = np.zeros((sample_size,stochastic_nvar))
    for v in range(stochastic_nvar):
        input_variable_array[:,v] = UQ_mapping_results.isel().get(model_param_varnames[v])
    #
    output_variable_array = np.zeros((sample_size,selected_mode_number))
    output_variable_array[:,:] = stochastic_coefficients[:,:]

    modes_Sobol_indices_1stOrder = np.zeros((stochastic_nvar,selected_mode_number))
    modes_Sobol_indices_Total = np.zeros((stochastic_nvar,selected_mode_number))
    #########Fit PCE with chosen parameters############
    for i in range(selected_mode_number):
        polynomialChaosResult = construct_PCE_ot(input_variable_array, \
                                                output_variable_array[:,i], marginals, \
                                                copula, PCE_deg, LARS=True) #LARS=True if sparse
        chaosSI = ot.FunctionalChaosSobolIndices(polynomialChaosResult)

        #########Get Sobol indices############
        modes_Sobol_indices_1stOrder[:,i] = [chaosSI.getSobolIndex(i) for i in range(stochastic_nvar)]
        modes_Sobol_indices_Total[:,i] =  [chaosSI.getSobolTotalIndex(i) for i in range(stochastic_nvar)]
        #print("Mode "+str(i),modes_Sobol_indices_1stOrder[:,i])

    aggregated_Sobol_indices_1stOrder = np.zeros((stochastic_nvar))
    aggregated_Sobol_indices_Total = np.zeros((stochastic_nvar))
    for j in range(stochastic_nvar): 
        aggregated_Sobol_indices_1stOrder[j] = np.sum(modes_Sobol_indices_1stOrder[j,:]*EVR[:selected_mode_number])
        aggregated_Sobol_indices_Total[j] = np.sum(modes_Sobol_indices_Total[j,:]*EVR[:selected_mode_number])
    
    
    return aggregated_Sobol_indices_1stOrder, aggregated_Sobol_indices_Total 
