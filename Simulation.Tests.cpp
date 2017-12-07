#ifndef SIMULATION_TESTS_CPP
#define SIMULATION_TESTS_CPP

#include "generic.hpp"

#include "Simulation.hpp"

void Simulation::performTests(){
// Runs Tests if they are commented out and stops the execution of the application afterwards

        generic::print(" ----- TEST MODE -----");

        auto TESTDirectEqualUdVcalculateBmat = [&](){
        // Tests wether calculateBmat(int,int,UdV,int) is equal to calculateBmatNoUdV for udvRate=infinity
        // Works, however only up to 1e-10. They should be exactly equal! But what does that mean regarding doubles.
            const double abstol = 1e-10;

            UdV udv_direct;
            UdV udv;
            Mat B_direct = calculateBmatNoUdV(simParams.m,0);
            UdVDecompose(B_direct,udv_direct);
            calculateBmat(simParams.m,0,udv,simParams.m+10);
            Mat B = udv.product();

            generic::print_compare_Mat_abs(B,B_direct,abstol);

            // std::cout.precision(30);
            // std::cout.setf(std::ios::fixed);
            // Mat B1 = Mat({B(0,0)});
            // Mat B_direct1 = Mat({B_direct(0,0)});
            // B1.raw_print(std::cout, "B:");
            // B_direct1.raw_print(std::cout, "B_direct:");
        };

        auto TESTDirectEqualNonsenseUdVcalculateBmat = [&](){
        // Tests wether calculateBmat(int,int,UdV,int) is equal to calculateBmatNoUdV for udvRate=1
        // when we take the udv product at the end (which mixes singular values again!)
        // True up to abstol 1e-9

            const double abstol = 1e-9;

            UdV udv_direct;
            UdV udv;
            Mat B_direct = calculateBmatNoUdV(simParams.m,0);
            UdVDecompose(B_direct,udv_direct);
            calculateBmat(simParams.m,0,udv,1);
            Mat B = udv.product();

            generic::print_compare_Mat_abs(B,B_direct,abstol);

            // std::cout.precision(30);
            // std::cout.setf(std::ios::fixed);
            // Mat B1 = Mat({B(0,0)});
            // Mat B_direct1 = Mat({B_direct(0,0)});
            // B1.raw_print(std::cout, "B:");
            // B_direct1.raw_print(std::cout, "B_direct:");
        };

        auto TESTDirectUnequalUdVcalculateBmat = [&](){
        // Tests wether calculateBmat(int,int,UdV,int) is unequal to calculateBmatNoUdV for udvRate=1
        // They are equal for 1e-0 and unequal for 1e-1 abstol.
            const double abstol = 1e-1;

            UdV udv_direct;
            UdV udv;
            Mat B_direct = calculateBmatNoUdV(simParams.m,0);
            UdVDecompose(B_direct,udv_direct);
            calculateBmat(simParams.m,0,udv,1);

            generic::print_compare_UdV_abs(udv_direct,udv,abstol);

            // std::cout.precision(30);
            // std::cout.setf(std::ios::fixed);
            // Mat B1 = Mat({B(0,0)});
            // Mat B_direct1 = Mat({B_direct(0,0)});
            // B1.raw_print(std::cout, "B:");
            // B_direct1.raw_print(std::cout, "B_direct:");
        };

        auto TESTudvRateDependencecalculateBmatSV = [&](){
        // Tests for what critical value of udvRate B(l2,0) deviates in terms of singular values (within some tolerance)
        // from B(l2,0) calculated with udvRate=1
        // result for l2 = simParams.m: udvRate~108
            const double absExpdiff = 1; // Difference in a singular value of ~ one order of magnitude
            const int l2=simParams.m;

            generic::print_var(simParams.m);
            
            // Calculate reference Bmat (with udvRate=1)
            int udvRate=1;
            UdV udvRef; VecReal dRef;
            calculateBmat(l2,0,udvRef,1);
            dRef = udvRef.d;

            UdV udv;
            double max_deviation=0;
            VecReal dUDV;
            do{
                ++udvRate;
                calculateBmat(l2,0,udv,udvRate);
                dUDV = udv.d;

                max_deviation = 0;
                for(int idx=0;idx<dRef.n_rows;++idx){
                    // relative deviation
                    double deviation = std::abs(std::log(dRef(idx))-std::log(dUDV(idx)));
                    max_deviation = (deviation>max_deviation)?deviation:max_deviation;
                }
            }while(max_deviation<absExpdiff&&udvRate<simParams.m);

            generic::print("TESTudvRateDependencecalculateBmatSV:\n");
            generic::print("Found critical udvRate: ", udvRate);
            
            generic::print("\nSingular value comparison:");
            for(int k=0;k<dRef.n_rows;++k){
                generic::print("Diff of singular values (svUDV="+generic::num2str(dUDV(k))+", svRef="+generic::num2str(dRef(k))+"): ", std::abs(std::log(dRef(k))-std::log(dUDV(k))));
            }
        };

        auto TESTudvRateDependencecalculateBmat = [&](){
        // Tests for what critical value of udvRate B(l2,0) deviates in terms of matrix elements (within some tolerance)
        // from B(l2,0) calculated with udvRate=1
        // 
            const double abstol = 1e-0; // Difference in a singular value of ~ one order of magnitude
            const int l2=simParams.m;

            generic::print_var(simParams.m);
            
            // Calculate reference Bmat (with udvRate=1)
            int udvRate=1;
            UdV udvRef;
            calculateBmat(l2,0,udvRef,1);

            UdV udv;
            double max_deviation=0;
            do{
                calculateBmat(l2,0,udv,udvRate);
                ++udvRate;
            // TODO: THIS HAS BEEN CHANGED. RECONSIDER WHILE CONDITION
            }while(generic::compare_UdV_abs_max(udvRef,udv,abstol)<100000&&udvRate<=simParams.m);

            generic::print("TESTudvRateDependencecalculateBmat:\n");
            generic::print("Found critical udvRate: ", udvRate-1);
        };

        auto TESTDirectVSUdVcalculateBmat = [&](){
        // Tests for what critical l2 in B(l2,0) we have a deviation of singular values or eigenvalues (within some tolerance)
        // of B(l2,0) calculated either directly or using UdV decomposition
        // result for udvRate=1: l2=28 (OLD RESULT SHOULD BE TESTED AGAIN)
            const double abstol = 1e-05;
            
            int criticall2 = 0;
            Mat BmatDirect;
            UdV udv;
            UdV udv_direct;
            double max_deviation=0;
            VecReal dDirect; VecReal dUDV;
            do{
                ++criticall2;
                BmatDirect = calculateBmatNoUdV(criticall2,0);
                calculateBmat(criticall2,0,udv,1);
                dUDV = udv.d;
                UdVDecompose(BmatDirect,udv_direct);
                dDirect = udv_direct.d;

                max_deviation = 0;
                for(int idx=0;idx<dDirect.n_rows;++idx){
                    // relative deviation
                    double deviation = std::abs(dDirect(idx)-dUDV(idx));
                    max_deviation = (deviation>max_deviation)?deviation:max_deviation;
                }
                //generic::print("max_dev: ", max_deviation);
            }while(max_deviation<abstol&&criticall2<simParams.m);

            generic::print("TESTDirectVSUdVcalculateBmat:\n");
            generic::print("Found critical l2: ", criticall2);
            
            generic::print("\nSingular value comparison:");
            for(int k=0;k<dDirect.n_rows;++k){
                generic::print("RelDiff of singular values (svUDV="+generic::num2str(dUDV(k))+", svDirect="+generic::num2str(dDirect(k))+"): ", generic::relDiff(dUDV(k),dDirect(k)));
            }
        };

        auto TESTcalculateGFfromScratchDtauBeta = [&](){
        // Calculates the GF from scratch with dtau=beta. In this case, m=1 and G[l=1]= [1+B_1]^(-1) 
        // SHOULD BE TESTED AGAIN
            const double reltol = 1e-06;
            simParams.dtau = simParams.beta;
            simParams.s = 1;
            simParams.init();
            this->init();
            //writeConfigurationToFile("phi_dtau-beta.conf");
            generic::print_var(simParams.dtau);
            generic::print_var(simParams.m);
            generic::print_var(simParams.dn_tau);
            //generic::print("tausteps: ", RowVecReal(simParams.tausteps));
            
            Mat B = calculateBmat(1);
            Mat ginv = Eye+B;
            Mat g = arma::inv(ginv);

            Mat gUDV;
            UdV udvB;
            calculateBmat(1,0,udvB,1);
            greenFromUdV(gUDV,udvB);

            Mat gUDVinv = Eye+udvB.product();
            Mat gUDVinvinv1 = gUDVinv*gUDV;
            Mat gUDVinvinv2 = gUDV*gUDVinv; 
            generic::print_compare(gUDVinvinv1,Eye,reltol);
            generic::print_compare(gUDVinvinv2,Eye,reltol);

            generic::print_var(gUDVinvinv1);
            generic::print_var(gUDVinvinv2);

            Mat gscratch;
            calculateGFfromScratch(gscratch, 1);

            generic::print_compare(g,gscratch,reltol);
            generic::print_compare(g,gUDV,reltol);
            generic::print_compare(gUDV,gscratch,reltol);
        };

        auto TESTcalculateGFfromScratchVSupdateInSlice = [&](){
        // Compares GF calculated from scratch and updated in slice after single local update.
            const double abstol = 2e-01;

            // writeConfigurationToFile("phiField.conf");

            // Calculate from scratch before new local update
            Mat gBefore;
            // calculatePotentialExponential(1,false).save("eVOld",arma::raw_ascii);
            calculateGFfromScratch(gBefore,1);
            gBefore.save("g_before",arma::raw_ascii);
            g = gBefore;

            // Update in slice
            std::tuple<RowVecReal, int, int> newlocalphi = proposeLocalUpdateAtSite(0,1);
            calculateM(newlocalphi);
            calculateMInverseAndDet();
            setPhi(newlocalphi);
            // writeConfigurationToFile("phiFieldNew.conf");
            // calculatePotentialExponential(1,false).save("eVNew",arma::raw_ascii);
            // generic::print_var(Delta_i);
            // generic::print_var(M);
            // generic::print_var(Minverse);
            updateGFIterativelyInSlice(newlocalphi);
            Mat gAfterUpdate = g;
            gAfterUpdate.save("g_after",arma::raw_ascii);

            // Calculate GF from Scratch
            Mat gAfterScratch;    
            calculateGFfromScratch(gAfterScratch,1);
            gAfterScratch.save("g_after_scratch",arma::raw_ascii);

            // Compare
            generic::print_compare_Mat_abs(gBefore,gAfterScratch,abstol);
            generic::print_compare_Mat_abs(gBefore,gAfterUpdate,abstol);
            generic::print_compare_Mat_abs(gAfterUpdate,gAfterScratch,abstol);

            // generic::print_var(gBefore);
            // generic::print_var(gAfterUpdate);
            // generic::print_var(gAfterScratch);

            Mat gDiff = gAfterScratch-gAfterUpdate;
            generic::print_var(gDiff);
        };

        auto TESTcalculateGFfromScratchVSupdateInSliceLatticeSweep = [&](){
        // Compares GF calculated from scratch and updated in slice after potential local updates for all lattice sites in timeslice.
            const double abstol = 1e-00;

            // writeConfigurationToFile("phiField.conf");

            // Calculate from scratch before new local update
            Mat gBefore;
            // calculatePotentialExponential(1,false).save("eVOld",arma::raw_ascii);
            calculateGFfromScratch(gBefore,1);
            gBefore.save("g_before",arma::raw_ascii);
            g = gBefore;

            // Update in slice
            performSpatialLatticeSweep(1);
            Mat gAfterUpdate = g;
            gAfterUpdate.save("g_after",arma::raw_ascii);

            // Calculate GF from Scratch
            Mat gAfterScratch;    
            calculateGFfromScratch(gAfterScratch,1);
            gAfterScratch.save("g_after_scratch",arma::raw_ascii);

            // Compare
            MatReal gDiff = arma::abs(gAfterScratch-gAfterUpdate);
            generic::print_var(gDiff);

            generic::print_var(gDiff.min());
            generic::print_var(gDiff.max());

            generic::print_compare_Mat_abs(gBefore,gAfterScratch,abstol);
            generic::print_compare_Mat_abs(gBefore,gAfterUpdate,abstol);
            generic::print_compare_Mat_abs(gAfterUpdate,gAfterScratch,abstol);
        };

        auto TESTwrapGFToNextTimeslice = [&] () {
            const double abstol = 1e-05;

            Mat g1; Mat g2; Mat g2wrapped;
            calculateGFfromScratch(g1,1);
            calculateGFfromScratch(g2,2);
            g2wrapped = g1;
            wrapGFToNextTimeslice(g2wrapped, 1, MonteCarlo::UPSWEEP);

            MatReal g21Diff = arma::abs(g2-g1);
            generic::print_var(g21Diff);
            generic::print_var(g21Diff.min());
            generic::print_var(g21Diff.max());

            MatReal gDiffWrapping = arma::abs(g2-g2wrapped);
            generic::print_var(gDiffWrapping);
            generic::print_var(gDiffWrapping.min());
            generic::print_var(gDiffWrapping.max());

            generic::print_compare_Mat_abs(g1,g2,abstol);
            generic::print_compare_Mat_abs(g1,g2wrapped,abstol);
            generic::print_compare_Mat_abs(g2,g2wrapped,abstol);
        };

        // auto TESTgreenFromUdV = [&](){
        // // Test if greenFromUdV(g,udv_l,udv_r) == greenFromUdV(g,identity,udv_r)
        // // Result: works.
        //     const double reltol = 1e-10;
        //     UdV udv_l = UdV(MatrixSizeFactor*simParams.N);
        //     udv_l.clear(MatrixSizeFactor*simParams.N);
        //     UdV udv_r = UdVDecompose(calculateBmat(1));
        //     Mat g1; Mat g2;
        //     greenFromUdV(g1,udv_r);
        //     greenFromUdV(g2,udv_l,udv_r);
        //     generic::print_compare(g1,g2,reltol);
        // };

        // auto TESTBmatCalculation =[&](){
        //     const double reltol = 1e-5;
        //     Mat B1=calculateBmat(3,3);
        //     generic::print_compare_cpx(B1,Eye,reltol); // works

        //     Mat B2 = calculateBmat(1,0);
        //     Mat B3 = calculateBmat(1);
        //     generic::print_compare_cpx(B2,B3,reltol);

        //     Mat B4 = calculateBmat(4,3);
        //     Mat B5 = calculateBmat(4);
        //     generic::print_compare_cpx(B4,B5,reltol);
        // };

        auto TESTSingularValuesOfBChain = [&](){
        // Calculate singular values of B(m,0)=Product_l=1^m for various beta
        // works: After fixing some bugs, they now evolve on all scales as a function of beta
            std::ofstream myfile;
            std::ofstream myfileeig;
            myfile.open ("SingularValues.csv");
            myfileeig.open ("EigenValues.csv");
            //myfile << "a,b,c,\n";
            for(double beta=0.1; beta<=20; beta+=0.1){
                simParams.T=1.0/beta;
                initRNG();
                commonInit();
                initPhiField();
                UdV udv;
                calculateBmat(simParams.m,0,udv,1);
                //Mat B = calculateBmatNoUdV(simParams.m,0);
                //UdVDecompose(B,udv);
                Vec ev = arma::eig_gen(udv.product());
                std::string str = generic::num2str(simParams.beta)+std::string(",");
                std::string streig = generic::num2str(simParams.beta)+std::string(",");
                for(int k=0;k<udv.d.n_rows;++k){
                    str += generic::num2str(std::log(udv.d(k)))+",";
                    streig += generic::num2str(std::log(std::abs(ev(k))))+",";
                }
                myfile << str << "\n";
                myfileeig << streig << "\n";
                myfile.flush();
                myfileeig.flush();
            }
            myfile.close();
            myfileeig.close();
        };

        auto TESTSingularValuesOfBChainUDVRate = [&](){
        // Calculate lowest singular value of B(m,0)=Product_l=1^m for various beta
        // and varying udvRate to see for what max value of udvRate we get a reasonable result (compared to udvRate=1)
        // OLD result: for udvRate<=20 we get good results up to beta=20
        // compare to TESTudvRateDependencecalculateBmat
            std::ofstream myfile;
            myfile.open ("LowestSVvsBetaAsFuncOfs.csv");
            UdV udv;
            for(double beta=0.1; beta<=20;beta+=0.1){
                std::string str = generic::num2str(beta)+std::string(",");
                for(int udvRate = 1; udvRate<=simParams.m; udvRate+=5){
                    simParams.T=1.0/beta;
                    initRNG();
                    commonInit();
                    initPhiField();
                    calculateBmat(simParams.m,0,udv,udvRate);
                    generic::print_var(udv.d(udv.d.n_rows-1));
                    str += generic::num2str(std::log(udv.d(udv.d.n_rows-1)))+",";
                }
                myfile << str << "\n";
                myfile.flush();
            }
            myfile.close();
        };

        auto TESTgetRandom2DVectorBox = [&](){
        // Tests the distribution of random box vectors generated by getRandom2DVectorBox(double box_length).
            const int numberOfDraws = 1000000;
            MatReal vectors = MatReal(numberOfDraws,2);
            for(int k=0; k<numberOfDraws; ++k){
                RowVecReal vec = getRandom2DVectorBox(1.0);
                vectors(k,0) = vec(0);
                vectors(k,1) = vec(1);
            }

            vectors.save("notebooks/vectors",arma::raw_ascii);
            generic::print(std::string("Written ")+generic::num2str(numberOfDraws)+std::string(" random vectors to file 'vectors'."));
        };

        auto TESTsaveAndLoad = [&](){
            // generic::print(simParams.toString());
            // boost::random::mt19937_64 rngbefore = rng;
            
            // generic::print_var(phiField);

            CubeReal phibefore = phiField;
            CubeReal phiconf = phiField;
            calculateGFfromScratch(g,simParams.m);
            Mat gbefore = g;

            this->writeConfigurationToFile(phiconf, "mytest1.conf");
            this->writeStateToFile("mystate1.state");
            this->loadStateFromFile("mystate1.state");
            this->writeStateToFile("mystate2.state");

            CubeReal phiafter = phiField;
            Mat gafter = g;
            this->loadConfigurationFromFile("mytest1.conf", phiconf);
            this->writeConfigurationToFile(phiconf, "mytest2.conf");

            std::cout.precision(20);
            std::cout.setf(std::ios::fixed);
            phibefore.slice(1).raw_print(std::cout,"phibefore");
            phiafter.slice(1).raw_print(std::cout,"phiafter");

            generic::print("phi equal? ", (phibefore==phiafter).min());

            generic::print("g equal? ", (gbefore==gafter).min());

            // boost::random::mt19937_64 rngafter = rng;
            // generic::print("rngs equal:", rngbefore==rngafter);
            // generic::print(simParams.toString());
        };

        auto ComparePhiFieldOf2StateFiles = [&](std::string file1, std::string file2){
            // generic::print(simParams.toString());
            // boost::random::mt19937_64 rngbefore = rng;
            
            // generic::print_var(phiField);

            CubeReal phi1;
            CubeReal phi2;
            this->loadStateFromFile(file1);
            phi1 = phiField;
            this->loadStateFromFile(file2);
            phi2 = phiField;

            std::cout.precision(20);
            std::cout.setf(std::ios::fixed);
            phi1.slice(1).raw_print(std::cout,"phi1");
            phi2.slice(1).raw_print(std::cout,"phi2");

            generic::print("phi equal? ", (phi1==phi2).min());

            // boost::random::mt19937_64 rngafter = rng;
            // generic::print("rngs equal:", rngbefore==rngafter);
            // generic::print(simParams.toString());
        };

        auto TESTCheckerboard = [&](){
            boost::timer::auto_cpu_timer timerCB(6);
            boost::timer::auto_cpu_timer timerFull(6);

            generic::print("Testing basic checkerboard breakup");
            timerCB.start();
            Mat Kcb = this->calculateKineticExponentialCheckerboard(-1,simParams.dtau/2.0);
            timerCB.stop();
            
            timerFull.start();
            this->calculateKineticExponential();
            Mat K = this->kineticExponential;
            timerFull.stop();

            timerCB.report();
            timerFull.report();

            generic::print_var(generic::compare_Mat_rel_max(K,Kcb,0));
            generic::print_var(generic::compare_Mat_rel_mean(K,Kcb,0));
            generic::print_var(generic::compare_Mat_abs_mean(K,Kcb,0));
            generic::print_var(generic::compare_Mat_abs_max(K,Kcb,0));

            // generic::print_var_prec(arma::abs(K-Kcb),1e-3);

            // generic::print_Mat_info(K);
            
        };

        //TESTcalculateGFfromScratchDtauBeta();
        //TESTgreenFromUdV();
        //TESTBmatCalculation();

        //TESTSingularValuesOfBChain();
        //TESTSingularValuesOfBChainUDVRate();
        
        
        //TESTcalculateGFfromScratchVSupdateInSlice();
        //TESTcalculateGFfromScratchVSupdateInSliceLatticeSweep();
        //TESTwrapGFToNextTimeslice();
        //TESTgetRandom2DVectorBox();

        //TESTDirectVSUdVcalculateBmat();

        // TESTudvRateDependencecalculateBmat();

        //TESTDirectEqualUdVcalculateBmat();
        //TESTDirectEqualNonsenseUdVcalculateBmat();
        //TESTDirectUnequalUdVcalculateBmat();

        //TESTsaveAndLoad();

        //ComparePhiFieldOf2StateFiles("compare.state","final.state");

        TESTCheckerboard();

        exit(0);
};

#endif