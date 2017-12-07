#ifndef SIMULATION_CPP
#define SIMULATION_CPP

#include "Simulation.hpp"

// -------- Implementation --------

// ## Initializion

    void Simulation::commonInit(){
    // Initialization indep. of wether the Simulations has been
    // initialized from scratch (Simulation::init()) or resumed (Simulation::initResume()).
        simParams.init();
        Eye = arma::eye<Mat>(MatrixSizeFactor*simParams.N,MatrixSizeFactor*simParams.N);
        Ones = arma::ones<Mat>(MatrixSizeFactor*simParams.N, MatrixSizeFactor*simParams.N);
        Zeros = arma::zeros<Mat>(MatrixSizeFactor*simParams.N, MatrixSizeFactor*simParams.N);
        createNeighborTable();
        createTimeNeighborTable();
        calculateKineticExponential();
        this->M = ZeroSmall;
        this->Mdet = 0;
        this->Minverse = ZeroSmall;
        sweepDirection = MonteCarlo::DOWNSWEEP; // We always start with a downsweep!
    }

    void Simulation::init(){
    // Initializes the complete simulation from scratch
        initRNG();
        commonInit();
        initPhiField();
        if(simParams.noFermions)
            initGreensFunctionEmpty();
        else
            initGreensFunction();
        stage = MonteCarlo::THERMALIZATION;
        substage = (simParams.adaptRounds==0?MonteCarlo::NONE:MonteCarlo::ADAPTION);
        if(tests) performTests();
    };

    void Simulation::initResume(){
    // Initializes the complete simulation from resumed settings
    // (Is called within loadStateFromFile())
        this->resume=true;
        commonInit();
        calculateBosonicAction();
        stage = simParams.currsweep>simParams.thermalSweeps?MonteCarlo::EQUILIBRIUM:MonteCarlo::THERMALIZATION;
        substage = simParams.currsweep<=(simParams.adaptRounds*simParams.sweepsPerAdaptRound)?MonteCarlo::ADAPTION:MonteCarlo::NONE;
    };

    void Simulation::initGreensFunction(){
    // Initializes the Greensfunction g from scratch
        std::cout << "Initializing Greens function ..." << std::flush;
        //g = arma::zeros<Mat>(2*simParams.N,2*simParams.N);
        calculateGFfromScratch(g,simParams.m,simParams.dn_tau); // We always start with a DOWNSWEEP in which we need G(beta).
        std::cout << " done." << std::endl;
    };

    void Simulation::initGreensFunctionEmpty(){
    // Initializes the Greensfunction g to a 2*Nx2*N zero matrix
        std::cout << "Initializing empty Greens function (noFermions=true?) ..." << std::flush;
        g = arma::zeros<Mat>(2*simParams.N,2*simParams.N);
        std::cout << " done." << std::endl;
    };

    void Simulation::initPhiField(){
    // Random initialization of system configuration, i.e. boson order parameter field phi
    // and calculation of associated boson action
        this->phiField.set_size(simParams.N,2,simParams.tausteps.size());
        
        // Alternative initializion
        //phiField.imbue( [&]() { return globals::randu01(); } );   // Init order parameter cube randomly
        if(simParams.initRandomPhiField){
            std::cout << "Initializing OP field ..." << std::flush;
            for(int l=0; l<this->phiField.n_slices; ++l) {
                for (int row=0; row<this->phiField.n_rows; ++row){
                    setPhi(row,l,getRandom2DVector(simParams.initMaxPhiLength));
                }
            }
        } else {
            std::cout << "Initializing zero OP field ..." << std::flush;
            this->phiField.zeros();
        }

        // Calculate and store bosonic action associated with initialized system configuration
        calculateBosonicAction();

        std::cout << " done." << std::endl;
    };

    void Simulation::initRNG(){
        rng = boost::random::mt19937_64(this->seed);
    };


// ## Monte Carlo simulation

    void Simulation::run(){
    // Core of Monte Carlo simulation
        std::cout << "Monte Carlo simulation" << (resume?" (resumed)":"") << std::endl;

        if (stage == MonteCarlo::THERMALIZATION){
            std::cout << "\t Thermalization stage..." << std::endl;
            if(substage == MonteCarlo::ADAPTION)
                if(resume)
                    generic::print("\t\t -------------------- Adapting on (continued)");
                else
                    generic::print("\t\t -------------------- Adapting on");
        }
        else if (stage == MonteCarlo::EQUILIBRIUM)
            std::cout << "\t Equilibrium stage..." << std::endl;

        // MAIN MONTE CARLO LOOP
        for(simParams.currsweep; simParams.currsweep<=simParams.totalSweeps; ++simParams.currsweep) {

            // // Checking (can be removed at a later time)
            // if(simParams.currsweep%2)
            //     assert(sweepDirection==MonteCarlo::DOWNSWEEP); // currsweep is odd, hence we should be doing an DOWNSWEEP
            // else
            //     assert(sweepDirection==MonteCarlo::UPSWEEP); // currsweep is odd, hence we should be doing an UPSWEEP

            // switch stage after thermalSweeps
            if (simParams.currsweep == (simParams.thermalSweeps+1)) {
                stage = MonteCarlo::EQUILIBRIUM;
                substage = MonteCarlo::MEASUREMENTS;
                std::cout << "\t Equilibrium stage..." << std::endl;
            }   

            // console output
            if (simParams.currsweep%simParams.printRate == 0){
                if (stage == MonteCarlo::EQUILIBRIUM)
                    std::cout << "\t\t " << simParams.currsweep-simParams.thermalSweeps << std::endl;
                else
                    std::cout << "\t\t " << simParams.currsweep << std::endl;
            }

            if((substage==MonteCarlo::MEASUREMENTS) && ((simParams.currsweep-simParams.thermalSweeps)%simParams.measureRate==0)){
                initMeasurements();
                generic::print("\t\t -------------------- Measurement ("+generic::num2str(simParams.currsweep)+")");
            }

            if (sweepDirection == MonteCarlo::DOWNSWEEP) { // We completed two sweeps (DOWNSWEEP and UPSWEEP)
                setupUdVstorage();
            }
            double sweepAcceptRate = sweep(); // Perform space-time lattice sweep in one imaginary time direction

            if(measuring)
                finishMeasurements();

            // Adaption logic
            if (substage == MonteCarlo::ADAPTION){
                if(simParams.currsweep%simParams.sweepsPerAdaptRound==0){
                    double currAcceptRate = simParams.currAcceptRateSum/(double)simParams.sweepsPerAdaptRound;
                    //generic::print_var(currAcceptRate);
                    //generic::print_var(simParams.DeltaPhiLength);

                    // Increase or decrease acceptance rate by decreasing or increasing DeltaPhiLength
                    if(currAcceptRate<simParams.targetAcceptRate)
                        simParams.DeltaPhiLength*=(1.0-simParams.adaptPercentage);
                    else if(currAcceptRate>simParams.targetAcceptRate)
                        simParams.DeltaPhiLength*=(1.0+simParams.adaptPercentage);

                    simParams.currAcceptRateSum = 0.0;

                    // switch from substage ADAPTION to substage NONE if this was the last adaptRound
                    if ((simParams.currsweep/simParams.sweepsPerAdaptRound)==simParams.adaptRounds){    
                        substage = MonteCarlo::NONE;
                        generic::print("\t\t -------------------- Adapting off");
                        generic::print("\t\t                  |-- currAcceptRate = "+generic::num2str(currAcceptRate*100)+"%");
                        generic::print("\t\t                  |-- DeltaPhiLength = "+generic::num2str(simParams.DeltaPhiLength));

                        logger.acceptedProposalsCountDuringAdaption = logger.acceptedProposalsCount;
                        logger.proposalCountDuringAdaption = logger.proposalCount;
                    }
                }
                else
                    simParams.currAcceptRateSum += sweepAcceptRate; 
            }

            if ( fmod(simParams.currsweep/2.0,simParams.saveRate) == 0.0 ) {
                ++simParams.currsweep;
                assert(sweepDirection==MonteCarlo::DOWNSWEEP && "Only DOWNSWEEP saves allowed!!"); // We only save states, where the next sweep is DOWNSWEEP
                writeStateToFile("temp"+generic::num2str(tempStateCounter)+".state");
                --simParams.currsweep;
                if (simParams.keepTempStates)
                    ++this->tempStateCounter;
            }
        }

        timeSeriesToFile(times,normMeanPhiTimeSeries,"normMeanPhi.series");
        timeSeriesToFile(times,meanPhiTimeSeries,"meanPhi.series");
        timeSeriesToFile(times,meanNormPhiTimeSeries,"meanNormPhi.series");
        timeSeriesToFile(times,SDWSusceptTimeSeries,"SDWSuscept.series");

    };

    // ## Sweeps

        double Simulation::sweep(){
        // Performs a full space-time lattice sweep. GF is updated iteratively within sweeping spatial lattice
        // and from scratch every dn_tau timestep.
        // "sweep" := local proposal for each O(2) spin on every space-time lattice point (sweeping the lattice in one direction)
        // Returns the acceptance rate of the simParams.N * simParams.m local update proposals.

            using arma::trans; using arma::diagmat;

            double acceptRate = 0.0;
            if(sweepDirection == MonteCarlo::UPSWEEP)
            { // Going from tau=dtau to tau=beta

                // UdV storage at 0 contains B(beta,0) decomposition
                if(simParams.n!=0 && !simParams.noFermions){
                    greenFromUdV(g,UdVStorage.at(0)); // Calculate GF on time slice 0 from scratch based on UDV
                    UdVStorage.at(0).clear(MatrixSizeFactor*simParams.N); // Set 0 slot to identity = B(0,0)
                }

                for(int l = 1; l<=simParams.m; ++l)
                { // sum over timeslices
                    if(simParams.n!=0 && l%simParams.dn_tau==0 && !simParams.noFermions){ // Calculate GF from scratch based on UdVStorage 
                        int n_tau = l/simParams.dn_tau-1; // -1 to use same notation as in Assaad notes
                        // Mat gfwrapped = this->g;
                        advanceGFusingStorage(g, n_tau,MonteCarlo::UPSWEEP);
                        // Mat gfnew = this->g;
                        // double mean_reldiff = generic::compare_gf_mean(gfwrapped,gfnew);
                        // generic::print_var(mean_reldiff);
                    }

                    acceptRate += performSpatialLatticeSweep(l);
                    if(measuring) measure(l);
                    if(l!=simParams.m && !simParams.noFermions){
                        wrapGFToNextTimeslice(g, l, MonteCarlo::UPSWEEP);
                        
                    }

                }

                if(simParams.n!=0 && !simParams.noFermions) advanceGFusingStorage(g, simParams.n-1,MonteCarlo::UPSWEEP);

                sweepDirection = MonteCarlo::DOWNSWEEP; // Next time sweep from beta to dtau (MonteCarlo::DOWNSWEEP)

            }
            else
            { // MonteCarlo::DOWNSWEEP; Going from tau=beta to tau=dtau 

                // UdV storage at n contains B(beta,0) decomposition
                if(simParams.n!=0 && !simParams.noFermions){
                    // greenFromUdV(g,UdVStorage.at(simParams.n)); // Calculate GF on time slice beta from scratch based on UDV
                    UdVStorage.at(simParams.n).clear(MatrixSizeFactor*simParams.N); // Set n slot to identity = B(beta,beta)
                }
                for(int l = simParams.m; l>=1; --l)
                {

                    if( (simParams.n!=0&&l%simParams.dn_tau==0)&&(l!=simParams.m) && !simParams.noFermions){ // Upcoming timeslice is one of the simParams.n timeslices 
                        // Calculate GF of new time slice l from scratch based on UdVStorage
                        int n_tau = l/simParams.dn_tau+1; // +1 to use same notation as in Assaad notes
                        // Mat gwrapped = g; // DEBUG END
                        advanceGFusingStorage(g, n_tau,sweepDirection);
                    }

                    acceptRate += performSpatialLatticeSweep(l); // Within spatial lattice, GF is always updated iteratively
                    if(measuring) measure(l);
                    if(!simParams.noFermions)
                        wrapGFToNextTimeslice(g, l, MonteCarlo::DOWNSWEEP); // TODO: Could skip wrapping when next l is a n_tau timeslice

                }

                // Assaad notes say that we should sweep down from beta to dtau corresponding to l=m to l=1,
                // and thats exactly what we are doing. However, it then says storage should now contain UdV decompositions
                // for 0<=n_tau<=n. That is so far not the case, as we do not calculate the GF for n_tau=0 above
                // as l never reaches 0 which would lead to l/dn_tau=0.
                if(simParams.n!=0 && !simParams.noFermions) advanceGFusingStorage(g, 1,MonteCarlo::DOWNSWEEP);

                sweepDirection = MonteCarlo::UPSWEEP; // Next time sweep from dtau to beta (MonteCarlo::UPSWEEP)
            }

            return acceptRate/(double)simParams.m;
        };

        double Simulation::sweepNoUDVStorage(){
        // Performs a full space-time lattice sweep but not using any UDV Storage
        // (should in principle just be slower than sweep())
        // "sweep" := local proposal for each O(2) spin on every space-time lattice point (sweeping the lattice in one direction)
        // Returns the acceptance rate of the simParams.N * simParams.m local update proposals.

            using arma::trans; using arma::diagmat;

            double acceptRate = 0.0;
            if(sweepDirection == MonteCarlo::UPSWEEP)
            { // Going from tau=dtau to tau=beta

                for(int l = 1; l<=simParams.m; ++l)
                { // sum over timeslices

                    if(simParams.n!=0&&l%simParams.dn_tau==0 && !simParams.noFermions){ // Calculate GF from scratch 
                        calculateGFfromScratch(g, l);
                    }
                    acceptRate += performSpatialLatticeSweep(l);
                    if(measuring) measure(l);
                    if(l!=simParams.m && !simParams.noFermions)
                        wrapGFToNextTimeslice(g, l, MonteCarlo::UPSWEEP);
                }

                sweepDirection = MonteCarlo::DOWNSWEEP; // Next time sweep from beta to dtau (MonteCarlo::DOWNSWEEP)
            }
            else
            { // MonteCarlo::DOWNSWEEP; Going from tau=beta to tau=dtau 

                for(int l = simParams.m; l>=1; --l)
                {
                    if( (simParams.n!=0&&l%simParams.dn_tau==0)&&(l!=simParams.m) && !simParams.noFermions){ // Upcoming timeslice is one of the simParams.n timeslices 
                        // Calculate GF of new time slice l from scratch
                        calculateGFfromScratch(g, l);
                    }
                    acceptRate += performSpatialLatticeSweep(l); // Within spatial lattice, GF is always updated iteratively
                    if(measuring) measure(l);
                    if(!simParams.noFermions)
                        wrapGFToNextTimeslice(g, l, MonteCarlo::DOWNSWEEP); // TODO: Could skip wrapping when next l is a n_tau timeslice
                }

                sweepDirection = MonteCarlo::UPSWEEP; // Next time sweep from dtau to beta (MonteCarlo::UPSWEEP)
            }

            return acceptRate/(double)simParams.m;
        };

        double Simulation::performSpatialLatticeSweep(int timeslice){
        // Performs a sweep of local updates within one timeslice over all spatial lattice points
        // Returns how many (of the simParams.N) proposals have been accepted
            int acceptCounter = 0;
            for(int i=0;i<simParams.N;++i){
                std::tuple<RowVecReal,int,int> newlocalphi = proposeLocalUpdateAtSite(i,timeslice);
                bool accepted = acceptMetropolisLocal(newlocalphi);
                if (accepted)
                    ++acceptCounter;
            };
            return (double)acceptCounter/(double)simParams.N;
        };

    // ## Update proposals

        std::tuple<RowVecReal,int,int> Simulation::proposeLocalUpdateAtSite(int site, int tau){
        // Proposes a new configuration where on the given single site the order parameter vector is randomly
        // modified, i.e. rescaled and rotated. Returns tuple (newphi,site,tau)

            ++this->logger.proposalCount;
            RowVecReal deltaPhi;
            if(simParams.proposeMethod==SimulationParameters::BOXPROPOSE)
                deltaPhi = getRandom2DVectorBox(this->simParams.DeltaPhiLength);
            else // SimulationParameters::CIRCLEPROPOSE
                deltaPhi = getRandom2DVector(this->simParams.DeltaPhiLength);
            return std::make_tuple(getPhi(site,tau)+deltaPhi,site,tau);
        };

        std::tuple<RowVecReal,int,int> Simulation::proposeLocalUpdate(){
        // Receives the current configuration, i.e. order parameter field, and proposes
        // a new configuration with a random local change.
        // Returns a std::tuple with the new phi vector, the modified site and time.

            // Choose random site of space-time lattice and randomly choose a new phi vector
            int rdm_site = this->randint(0,phiField.n_rows-1);
            int rdm_tau = this->randint(0,phiField.n_slices-1);

            ++this->logger.proposalCount;
            return proposeLocalUpdateAtSite(rdm_site, rdm_tau);
        };



    // ## Calculate

        // ## Bosonic sector
            
            double Simulation::calculateBosonicAction(CubeReal& phiField){
            // Given a specific system configuration (bosonic order parameter field phi), the function calculates
            // the value of the bonsonic action S_phi.
            // The result is always >=0 

            // OPT: Could probably be vectorized, i.e. for spatial difference term shift lattice four times
            // and in each case substract from original phi field.
                double S_phi = 0.0;
                auto const dtau = simParams.dtau;

                for(int l=1; l<=simParams.m;++l){ // sum over imaginary time steps
                    for(int i=0; i<simParams.N; ++i){   // sum over sites

                        // temporal difference term
                        // using simple first order finite difference formula for tau derivative
                        int earlier = getTimeNeighbors(l).at(0);
                        RowVecReal derivative = getPhi(phiField,i,l)-getPhi(phiField,i,earlier);
                        S_phi += 1.0/(2.0*constants::c*constants::c*dtau)*arma::dot(derivative,derivative);

                        // spatial difference term
                        // Count only top and right neighbor (avoid overcounting)
                        uVecInt neighbors = getTopAndRightNeighbors(i);
                        for(int nn=0; nn<neighbors.size(); ++nn) {  // sum over neighbors
                            RowVecReal phidiff = getPhi(phiField,i,l) - getPhi(phiField,neighbors(nn),l); // phi difference between neighboring lattice sites
                            S_phi += dtau*1.0/2.0*arma::dot(phidiff,phidiff);
                        }

                        // mass term & quartic interaction
                        double phisquared = arma::dot(getPhi(phiField,i,l),getPhi(phiField,i,l));
                        S_phi += dtau*modelParams.r/2.0*phisquared;
                        S_phi += dtau*modelParams.u/4.0*phisquared*phisquared;
                    }
                }

                //arma::cube phisquared = conf.phi % conf.phi; // element-wise cube multiplication
                //arma::cube tmp = modelParams.r/2.0 * phisquared ;
                //phitmp.for_each( [](arma::cube::elem_type& elem) { elem += 123.0; } );  // NOTE: the '&' is crucial!

                assert(S_phi>=0.0);
                return S_phi;
            };
            double Simulation::calculateBosonicAction(){
                this->S_phi = calculateBosonicAction(this->phiField); 
                return this->S_phi;
            };

            double Simulation::calculateBosonicActionChange(std::tuple<RowVecReal,int,int>& newlocalphi){
            // Calculates DeltaS_phi = S_phi' - S_phi (after update minus before update) that is
            // the change of the bosonic action due to a proposed local update (newlocalphi)

                // Unpack local update information
                RowVecReal newphi;
                int site; int timeslice;
                std::tie(newphi,site,timeslice) = newlocalphi;
                RowVecReal oldphi = getPhi(site,timeslice);

                using arma::dot;

                RowVecReal phiDiff = newphi - oldphi;

                double oldphiSq = dot(oldphi, oldphi);
                double newphiSq = dot(newphi, newphi);
                double phiSqDiff = newphiSq - oldphiSq;

                double oldphiPow4 = oldphiSq * oldphiSq;
                double newphiPow4 = newphiSq * newphiSq;
                double phiPow4Diff = newphiPow4 - oldphiPow4;

                uVecInt tnn = getTimeNeighbors(timeslice);
                RowVecReal phiEarlier = getPhi(site,tnn(0));
                RowVecReal phiLater = getPhi(site,tnn(1));
                RowVecReal phiTimeNeigh = phiLater + phiEarlier;

                uVecInt nn = getNeighbors(site);
                RowVecReal phiSpaceNeigh = arma::zeros<RowVecReal>(2);
                for(int k=0;k<nn.size();++k)
                    phiSpaceNeigh += getPhi(nn(k),timeslice);

                const double dtau = simParams.dtau;
                const double r = modelParams.r;
                const double u = modelParams.u;
                const double c = constants::c;

                double dS_1 = (1.0 / (c * c * dtau)) * (phiSqDiff - dot(phiTimeNeigh, phiDiff));

                double dS_2 = 0.5 * dtau * (4 * phiSqDiff - 2.0 * dot(phiSpaceNeigh, phiDiff));

                double dS_3 = dtau * (0.5 * r * phiSqDiff + 0.25 * u * phiPow4Diff);

                return dS_1+dS_2+dS_3;

            };

        // ## Fermion sector (e.g. Green's function)

            std::tuple<double,double> Simulation::calculateCoshSinhTerm(int site, int timeslice, CubeReal& phiField, bool sign=false) {
            // Computes cosh and sinh expressions for the phi vector at site and timeslice of given
            // order parameter configuration. Returns the result as tuple (cosh(...), sinh(...))
                return Simulation::calculateCoshSinhTerm(getPhi(phiField,site,timeslice),sign);
            };

            std::tuple<double,double> Simulation::calculateCoshSinhTerm(RowVecReal phi, bool sign=false) {
            // Computes cosh and sinh expressions for the given phi vector
            // Returns the result as tuple (cosh(...), sinh(...)) 
                const double dtau = sign?(-1)*simParams.dtau:simParams.dtau;
                double phiNorm = arma::norm(phi, 2);
                return std::make_tuple(std::cosh(modelParams.lambda * dtau * phiNorm),
                                    std::sinh(modelParams.lambda * dtau * phiNorm) / phiNorm);
            };

            Mat Simulation::calculatePotentialExponential(CubeReal& phiField, int timeslice, bool sign=false){
            // Calculates e^(sign*dtau V(l)) for given system configuration and time slice. (sign: true means plus)
            // (We are only considering the xup,ydown sector and thus only returning this sector (V_tilde)
            // of the full potential exponential.)

                assert(timeslice>0);
                assert(timeslice<=simParams.m);

                // Shortcuts
                const int N = simParams.N;

                // Calculate C and S submatrices
                Vec C_diag_elem = arma::zeros<Vec>(N);
                Vec S_diag_elem = arma::zeros<Vec>(N);
                for (int j=0; j<phiField.n_rows; ++j){
                    double coshterm; double sinhterm;
                    std::tie(coshterm, sinhterm) = calculateCoshSinhTerm(j,timeslice,phiField,sign);
                    RowVecReal phivec = getPhi(phiField,j,timeslice);
                    C_diag_elem(j) = coshterm;
                    S_diag_elem(j).real(-1*phivec(0)*sinhterm);
                    S_diag_elem(j).imag(phivec(1)*sinhterm);
                }
                Mat C = arma::diagmat(C_diag_elem);
                Mat S = arma::diagmat(S_diag_elem);
                /*std::cout << "########" << std::endl << std::endl;
                std::cout << C << std::endl << std::endl;
                std::cout << S << std::endl << std::endl;*/

                // Construct exponential of potential (blockmatrix)
                // OPT: Should be a sparse matrix
                Mat V_tilde = arma::zeros<Mat>(2*N,2*N);
                generic::block(V_tilde,2,0,0)=C;
                generic::block(V_tilde,2,0,1)=S;
                generic::block(V_tilde,2,1,0)=arma::conj(S);
                generic::block(V_tilde,2,1,1)=C;

                // Calculate full eV (not only upper left sector)
                /*                Cube blocks = arma::zeros<Cube>(2*N,2*N,4);
                blocks.slice(0) = V_tilde;
                blocks.slice(3) = arma::conj(V_tilde);*/
                //Mat eV = arma::zeros<Mat>(4*N,4*N); // N lattice sites, 2 flavours, 2 spin
                /* Mat eV = generic::blockmat(blocks);*/

                return V_tilde;
            };
            Mat Simulation::calculatePotentialExponential(int timeslice, bool sign=false){
                return calculatePotentialExponential(this->phiField, timeslice, sign);
            };

            void Simulation::calculateKineticExponential(){
            // Calculates e^(-dtau/2 K) for given system configuration and stores it in attribute kineticExponential.
            // (We are only considering the xup,ydown sector and thus only storing this sector
            // of the full kinetic exponential.)
                // Shortcuts
                unsigned int N = simParams.N;
                double dtau = simParams.dtau;

                // Need t_x_up_h, t_x_down_h, t_x_up_v, t_x_down_v, t_y_up_h, t_y_down_h, t_y_up_v, t_y_down_v
                Cube t = hoppingCube();

                // Outcommented, because we only need Kxup and Kydown!
                // Construct NxN kinetic matrices, that is [K]_ij for each configuration of spin (up, down) and
                // flavor (x,y). Store those 4 matrices in 2x2 arma::field (row = spin, col = flavor)
                // arma::field<Mat> K(2,2);
                // for(int spin=0; spin<2; ++spin){
                //     for(int flav=0; flav<2; ++flav){
                //         Mat Temp = arma::zeros<Mat>(N,N);
                //         Temp.diag() += modelParams.mu; // Chemical potential mu on diagonal
                //         for(int col=0; col<N; ++col) {
                //             uVecInt nn = getNeighbors(col);
                //             Temp(nn(0),col) =	t(spin,1,flav); // up nn
                //             Temp(nn(1),col) =   t(spin,1,flav); // down nn
                //             Temp(nn(2),col) =   t(spin,0,flav); // left nn
                //             Temp(nn(3),col) =   t(spin,0,flav); // right nn
                //         }
                //         K(spin,flav) = (-1)*Temp;
                //     }        
                // }

                Mat Kxup = arma::zeros<Mat>(N,N);
                Mat Kydown = arma::zeros<Mat>(N,N);
                Kxup.diag() += modelParams.mu; // Chemical potential mu on diagonal
                Kydown.diag() += modelParams.mu; // Chemical potential mu on diagonal
                for(int col=0; col<N; ++col) {
                    uVecInt nn = getNeighbors(col);
                    Kxup(nn(0),col) =	t(0,1,0); // up nn
                    Kxup(nn(1),col) =   t(0,1,0); // down nn
                    Kxup(nn(2),col) =   t(0,0,0); // left nn
                    Kxup(nn(3),col) =   t(0,0,0); // right nn

                    Kydown(nn(0),col) =	t(1,1,1); // up nn
                    Kydown(nn(1),col) = t(1,1,1); // down nn
                    Kydown(nn(2),col) = t(1,0,1); // left nn
                    Kydown(nn(3),col) = t(1,0,1); // right nn
                }
                Kxup = (-1)*Kxup;
                Kydown = (-1)*Kydown;

                // Calculate matrix exponentials and block diagonal kinetic exponential
                // Using arma::expmat for matrix exponentials. Alternative: diagonalize and apply exp() to diag
                Cube blocks = Cube(N,N,2);
                blocks.slice(0) = arma::expmat(-dtau/2.0*Kxup);
                blocks.slice(1) = arma::expmat(-dtau/2.0*Kydown);
                /*blocks.slice(2) = arma::expmat(-dtau/2.0*K(1,0));
                blocks.slice(3) = arma::expmat(-dtau/2.0*K(0,1));*/
                this->kineticExponential = generic::blockdiagmat(blocks);
            };

            Mat Simulation::calculateBmat(int l){
            // Calculates B_tilde(l) matrix, where l = tau/deltaTau.
            // (We are only considering the xup,ydown sector and thus returning this sector only
            // (B_tilde instead of B).)
                
                assert(l>0);
                assert(l<=simParams.m);

                Mat eV = calculatePotentialExponential(l,false);
                return kineticExponential*eV*kineticExponential;
            };

            Mat Simulation::calculateBmatNoUdV(int l2, int l1){
            // Naive calculation of B(l2,l1) = product of B matrices, i.e. naive direct calculation of B matrices product\
            // If l2-l1 is too large, small singular values will get lost due to roundoff errors on finite precision machines.
            // WARNING: This function should only be used for small values of |l2-l1|!
                assert(l2 >= l1);
                assert(l2 <= simParams.m);
                assert(l1 >= 0);
                
                if(l1==l2) return Eye; // No propagation for equal times.
                if(l1==l2-1) return calculateBmat(l2); // Shortcut for single B factor

                Mat product = Eye;
                for(int l=l1+1; l<=l2;++l){
                    product = calculateBmat(l)*product; // later times to the left! Not clear from Definition in Assaad 2008 Eq. (103)!
                }
                return product;
            };

            void Simulation::calculateBmat(int l2, int l1, UdV& udv_res, int udvRate){
            // Calculate the imaginary time propagator B(tau2,tau1) for discretized time (tau=l*Delta_tau) as definied in Assaad 2008 Eq. (103)
            // corresponding to a product of B(l) matrices where l runs from l1+1 to l2 (later times to the left).
            // The chain of product of B matrices is made stable as described in Max, Erez and Assaad notes by only calculating
            // products involving no more than udvRate B matrices directly.
            // We follow Max notes around Eq. (76). However, this only really kicks in, if udvRate<l2-l1.
            //
            // REMARK: Here udvRate specifies the maximal length of a product of B matrices. Hence, UDV decomposition steps
            // are, in general, NOT commensurable with the fixed n_tau timeslices where we calculate GF from scratch during sweep.
            //
            // APPLICATION: For udvRate=1 this gives the best (exact) solution for B(l2,l1) that we have. Hence this might be used for comparison.
                assert(l2 >= l1);
                assert(l2 <= simParams.m);
                assert(l1 >= 0);
                assert(udvRate>0);

                if(l1==l2){ udv_res.clear(MatrixSizeFactor*simParams.N); return; } // No propagation for equal times.
                if(l1==l2-1){ udv_res=UdVDecompose(calculateBmat(l2)); return; } // Never use UDV decomposition for single B factor.

                auto integrateProductIntoUDV = [] (Mat product, UdV& udv_l) {
                // Takes a product chain of B matrices that could be calculated stably (# of B matrices <= udvRate) and a UDV decomp.
                // of a (previous) B-matrix product and saves the UDV decomp. of the whole product into udv_l. 
                        Mat partial_prod = product*udv_l.U;
                        partial_prod = partial_prod * arma::diagmat(udv_l.d);
                        Mat V_t_temp;
                        UdVDecompose(partial_prod,udv_l.U,udv_l.d,V_t_temp);
                        udv_l.V_t = udv_l.V_t*V_t_temp;
                    };

                Mat product = Eye;               
                udv_res.clear(MatrixSizeFactor*simParams.N); // SVD decomposition of temporary result
                bool naiveProduct = true;
                for(int l=l1+1; l<=l2;++l){
                    if((l-l1)%udvRate==0)
                    {   // B product gets too long -> UDV decomposition, start new product
                        integrateProductIntoUDV(product,udv_res);
                        product = calculateBmat(l);
                        naiveProduct=false;
                    }
                    else
                    {   // Calculate naive product
                        product = calculateBmat(l)*product; // later times to the left! Not clear from Definition in Assaad 2008 Eq. (103)!
                    }
                }

                if(!naiveProduct){
                    integrateProductIntoUDV(product,udv_res);
                    return;
                }
                UdVDecompose(product,udv_res);
            };

            Mat Simulation::calculateGFfromScratchNaive(Mat& g, int l){
            // Calculates fermionic equal-time Green's function g(l)=G_phi_tilde(l)
            // Assaad 2008 Eq. (119)
            // WARNING: Probably never works, as small singular values get lost in B mat products due to
            //          roundoff errors.
                Mat g_inv = Eye;
                g_inv = g_inv + calculateBmatNoUdV(l,0)*calculateBmatNoUdV(simParams.m,l);

                try{
                    return arma::inv(g_inv);
                }
                catch(const std::exception& e){
                    std::cout << "In calculateGFfromScratchNaive(), inversion of g_inv failed (probably singular): " << e.what() << std::endl;
                }

                return Zero;
            };

            void Simulation::calculateGFfromScratch(Mat& g, int l, int udvRate){
            // Calculate the fermionic equal-time Green's function from scratch based on UDV decomposition
            // (but NOT using the UdVStorage!)

                assert(l>=0);
                assert(l<=simParams.m);
                if(udvRate<=0) udvRate=simParams.m+10; // Dont use udv in B chain product

                UdV udv_l; UdV udv_r;
                if(l==simParams.m||l==0) // g(0) = g(m)!
                {
                    calculateBmat(simParams.m,0,udv_r,udvRate);
                    greenFromUdV(g,udv_r);
                }
                else
                {
                    calculateBmat(simParams.m,l,udv_l,udvRate);
                    calculateBmat(l,0,udv_r,udvRate);
                    greenFromUdV(g,udv_l,udv_r);
                }
            };

            void Simulation::advanceGFusingStorage(Mat& g, int n_tau, MonteCarlo::SweepDirection sd){
            // This function is called every dn_tau timestep and calculates
            // the equal time Greens function on n_tau+-1 according to SweepDirection from scratch based on the UdV Storage.
            // The result for the GF is stored in attribute g.
            // WARNING: Compatible with Assaad notes 2008 pp.61-62, the index n_tau specifies the last timeslice where we already
            //          calculated the GF from scratch. Hence, calling this function will actually calculate GF at n_tau-1 (DOWNSWEEP)
            //          or n_tau+1 (UPSWEEP) and fill corresponding UdV storage slot.
                assert(simParams.n!=0);

                if(sd==MonteCarlo::DOWNSWEEP)
                {
                    // Read B((n_tau-1)tau1,0) from (old) storage
                    UdV& udv_R = UdVStorage.at(n_tau-1);

                    // Compute B(n_tau*tau1,(n_tau-1)tau1)
                    Mat B_tau_tauprev = calculateBmatNoUdV(simParams.n_tau2l(n_tau),simParams.n_tau2l(n_tau-1));

                    // Read B(beta,n_tau*tau1)
                    Mat& U_tilde_L = UdVStorage.at(n_tau).U;
                    VecReal& d_tilde_L = UdVStorage.at(n_tau).d;
                    Mat& V_t_tilde_L = UdVStorage.at(n_tau).V_t;

                    // Calculate B(beta,(n_tau-1)tau1) via Eq. (149)
                    UdV udv_L; // will contain UdV decomposition of the B matrix (called udv_2 in Eq. (149))
                    Mat temp = trans(V_t_tilde_L)*B_tau_tauprev;
                    UdVDecompose(diagmat(d_tilde_L)*temp, udv_L);
                    udv_L.U = U_tilde_L * udv_L.U;  
                    
                    // Calculate GF
                    greenFromUdV(g,udv_L,udv_R);

                    // Update n_tau-1 slot of storage with B(beta,(n_tau-1)tau1)
                    UdVStorage.at(n_tau-1).U = udv_L.U;
                    UdVStorage.at(n_tau-1).d = udv_L.d;
                    UdVStorage.at(n_tau-1).V_t = udv_L.V_t;
                }
                else
                { // MonteCarlo::UPSWEEP
                    
                    // Read B(beta,(n_tau+1)tau1) from (old) storage
                    UdV& udv_L = UdVStorage.at(n_tau+1);

                    // Compute B((n_tau+1)tau1,n_tau*tau1)
                    Mat B_taunext_tau = calculateBmatNoUdV(simParams.n_tau2l(n_tau+1),simParams.n_tau2l(n_tau));

                    // Read B(n_tau*tau1,0)
                    Mat& U_tilde_R = UdVStorage.at(n_tau).U;
                    VecReal& d_tilde_R = UdVStorage.at(n_tau).d;
                    Mat& V_t_tilde_R = UdVStorage.at(n_tau).V_t;

                    // Calculate B((n_tau+1)tau1,0) via Eq. (149)
                    UdV udv_R; // will contain UdV decomposition of the B matrix (called udv_2 in Eq. (149))
                    Mat temp = B_taunext_tau*U_tilde_R;
                    UdVDecompose(temp*diagmat(d_tilde_R), udv_R);
                    udv_R.V_t = V_t_tilde_R * udv_R.V_t;                          

                    // Calculate GF
                    greenFromUdV(g,udv_L,udv_R);

                    // Update n_tau+1 slot of storage with B((n_tau+1)tau1,0)
                    UdVStorage.at(n_tau+1).U = udv_R.U;
                    UdVStorage.at(n_tau+1).d = udv_R.d;
                    UdVStorage.at(n_tau+1).V_t = udv_R.V_t;
                }
            };

            void Simulation::greenFromUdV(Mat& green_out, const UdV& UdV_l, const UdV& UdV_r) const{
            // Calculates the fermionic equal-time Green's function g in a numerically stable way using UdV decomp. of B matrices
            // Fast version with no access to time-displaced Green functions. (Assaad 2008 p.53 very bottom)
            // Recall, that unitarity of X implies X^{-1}=X^dagger=X.t()
            // Uses B(beta, tau) = U_l d_l V_l   and     B(tau, 0) = U_r d_r V_r,
            // computes G(tau) = [Id + B(tau,0).B(beta,tau)]^{-1}
            //                 = [Id + U_r d_r V_r U_l d_l V_l]^{-1}
            //                 = (V_t_L V_t_x) D_x^{-1} (U_R U_x)^{dagger}
                const Mat&   U_l   = UdV_l.U;
                const VecReal& d_l   = UdV_l.d;
                const Mat&   V_t_l = UdV_l.V_t;
                const Mat&   U_r   = UdV_r.U;
                const VecReal& d_r   = UdV_r.d;
                const Mat&   V_t_r = UdV_r.V_t;

                using arma::diagmat; using arma::trans;

                Mat VU_rl_product = trans(V_t_r) * U_l;
                Mat UtVt_rl_product = trans(U_r) * V_t_l;

                // UdV decomposition of "middle part" of G^{-1}
                Mat U_temp, V_t_temp; VecReal green_inv_sv;
                UdVDecompose(UtVt_rl_product + diagmat(d_r) * VU_rl_product * diagmat(d_l),U_temp, green_inv_sv, V_t_temp);                    
                
                Mat Vt_product = V_t_l * V_t_temp;
                Mat U_product  = U_r * U_temp;
                
                green_out = Vt_product *
                    diagmat(1.0 / green_inv_sv) *
                    trans(U_product);
            }

            void Simulation::greenFromUdV(Mat& green_out, const UdV& UdV_r) const {
            // Special case tau=beta where B(beta,tau)=B(beta,beta)=1=U_l*D_l*V_l.
            // OR case tau=0 where B(tau,0)=B(0,0)=1
            // In both cases, the product B(tau,0)B(beta,tau)=B(beta,0)
    
                const Mat&   U_r   = UdV_r.U;
                const VecReal& d_r   = UdV_r.d;
                const Mat&   V_t_r = UdV_r.V_t;

                // generic::print("U_r: ", U_r);
                // generic::print("d_r: ", d_r);
                // generic::print("V_t_r: ", V_t_r);  

                using arma::diagmat; using arma::trans;

                Mat U_temp, V_t_temp; VecReal green_inv_sv;
                UdVDecompose(trans(U_r) * V_t_r + diagmat(d_r), U_temp, green_inv_sv, V_t_temp);          

                // generic::print("mattemp: ", trans(U_r) * V_t_r + diagmat(d_r));
                // generic::print("U_temp: ", U_temp);
                // generic::print("d_temp: ", green_inv_sv);
                // generic::print("V_t_temp: ", V_t_temp);          

                Mat V_t_product = V_t_r * V_t_temp;
                Mat U_product = U_r * U_temp;
                
                green_out = V_t_product *
                        diagmat(1.0 / green_inv_sv) *
                            trans(U_product);
            }

            void Simulation::updateGFIterativelyInSlice(std::tuple<RowVecReal,int,int>& newlocalphi){
            // Calculate fermionic GF of new configuration differing by single local change based on old GF,
            // i.e. NOT from scratch but iteratively!

                unsigned int site = std::get<1>(newlocalphi);
                Mat gpart1 = Mat(2*simParams.N,2);
                Mat gMinusOnepart2 = Mat(2,2*simParams.N);
                gpart1 = g.submat(arma::regspace<arma::uvec>(0,2*simParams.N-1),arma::uvec({site,site+simParams.N}));
                Mat temp = g-Eye;
                gMinusOnepart2 = temp.submat(arma::uvec({site,site+simParams.N}),arma::regspace<arma::uvec>(0,2*simParams.N-1));

                Mat part1 = gpart1 * Delta_i;
                Mat part2 = Minverse * gMinusOnepart2; 
                this->g = g + part1*part2;

            };

            void Simulation::wrapGFToNextTimeslice(Mat& g, int timeslice, MonteCarlo::SweepDirection sd){
            // Given that g is the equal time GF at time slice timeslice, this function wraps the GF to the next time slice
            // depending on current direction of sweep.
                if(sd == MonteCarlo::DOWNSWEEP){
                    // Wrap GF to timeslice-1
                    g = arma::inv(calculateBmat(timeslice))*g*calculateBmat(timeslice);
                }
                else
                { // MonteCarlo::UPSWEEP
                    // Wrap GF to timeslice+1
                    g = calculateBmat(timeslice+1)*g*arma::inv(calculateBmat(timeslice+1));
                }
            };

            // -- Checkerboard decomposition --

            //Mat Simulation::leftMultiplyKineticExponentialCheckerboard(Mat& A){
            //// Takes a 2Nx2N matrix A (e.g. potential exponential eV) and multiplies
            //// e^-dtauK1/2*e^-dtauK2/2 from the left to it, where K1 and K2 represent the
            //// checkerboard decomposition.
            //    // TODO
            //};


            // -- UdV Storage --

            void Simulation::calculateUdVofBmat(int n_tau){
            // Implementation of Eq. (149) in 2008 Assaad notes to update next UdV slot in UdVStorage, that is UDV decomposition of B(n_tau,0),
            // based on previous UdV, that is UdVStorage.at(n_tau-1) = UDV decomposition of B(n_tau-1,0)
            // Note: B(n_tau,0)=B(n_tau,n_tau-1)*B(n_tau-1,0)=((B(n_tau,n_tau-1)*U_n_tau-1)*D_n_tau-1)*V_n_tau-1

                assert((n_tau-1)>0);
                assert(n_tau<=simParams.n);

                Mat& U = UdVStorage.at(n_tau-1).U;
                VecReal& d = UdVStorage.at(n_tau-1).d;
                Mat& V_t = UdVStorage.at(n_tau-1).V_t;

                Mat temp = calculateBmatNoUdV(simParams.n_tau2l(n_tau),simParams.n_tau2l(n_tau-1))*U;
                temp = temp * arma::diagmat(d);
                UdVDecompose(temp,UdVStorage.at(n_tau));
                UdVStorage.at(n_tau).V_t=V_t*UdVStorage.at(n_tau).V_t; // V_2 = V*V_1 -> V_2_t = V_1_t*V_t , where _t indicates conjugate-transpose
                // UdVStorage.at(n_tau) now contains UDV decomposition of B(n_tau,0)
            };

            void Simulation::setupUdVstorage(){
            // Set's up the UdV storage and initializes the decompositions as described before Eq. (178) in Assaad 2008
            // Afterwards, the UdVStorage contains UDV decompositions of all B(n_tau,0), 1<=n_tau<=n
                if(simParams.n!=0 && !simParams.noFermions){
                    UdVStorage.resize(simParams.n+1); // Maybe better to only reserve() and use pushback instead of at()
                    Mat B = calculateBmatNoUdV(simParams.n_tau2l(1),0); // B(tau1,0) = U_1 D_1 V_1
                    UdVDecompose(B,UdVStorage.at(1)); 
                    UdVStorage.at(0).clear(MatrixSizeFactor*simParams.N); // B(0,0) = 1; Not in Assaad notes but seems to be necessary!
                    for(int n_tau = 2; n_tau<=simParams.n; ++n_tau){
                        calculateUdVofBmat(n_tau);
                        //calculateBmat(simParams.n_tau2l(n_tau),0,UdVStorage.at(n_tau));
                    }
                }
            };

        // ## Acceptance probability

            Mat Simulation::calculateDeltaNaive(int l, CubeReal& phiField){
            // Calculates Delta as defined below (A18) in Max, Yoni paper
            // Full, direct calculation. Better only use for checking.

                Mat delta = calculatePotentialExponential(phiField,l,false) * calculatePotentialExponential(this->phiField,l,true);
                delta = delta - Eye;
                return delta;
            };

            void Simulation::calculateDelta_i(std::tuple<RowVecReal,int,int>& newlocalphi){
            // Computes the non-zero elements of sparse matrix Delta, that is Delta_i.
            // Delta = e^(-dtau*V_new)*e^(+dtau*V_old) - 1
            //
            // Based on Max: get_delta_forsite()
                int site;int timeslice; RowVecReal newphi;
                std::tie(newphi,site, timeslice) = newlocalphi;
            
                //eVMatrix(): yield a 4x4 matrix containing the entries of e^(sign*dtau*V) ~ V_tilde_sign
                // for the given lattice site and time slice (which e.g. has been changed in local proposal).
                auto eVMatrix = [this](
                    int sign,
                    RowVecReal kphi, // Cartesian!
                    double kcoshTermPhi, double ksinhTermPhi) -> MatSmall
                    {
                        MatSmall eV;
                        
                        Simulation::setReal(eV(0,0), kcoshTermPhi);
                        Simulation::setReal(eV(1,1), kcoshTermPhi);
                        Simulation::setReal(eV(0,1), sign * kphi[0] * ksinhTermPhi);
                        Simulation::setReal(eV(1,0), sign * kphi[0] * ksinhTermPhi);
                        
                        Simulation::setImag(eV(0,1),-sign * kphi[1] * ksinhTermPhi);
                        Simulation::setImag(eV(1,0),sign * kphi[1] * ksinhTermPhi);
                        return eV;
                    };

                double coshold;double sinhold;
                double coshnew;double sinhnew;
                std::tie(coshold,sinhold) = calculateCoshSinhTerm(site,timeslice,this->phiField);
                std::tie(coshnew,sinhnew) = calculateCoshSinhTerm(newphi);

                MatSmall eVOld = eVMatrix(
                                    +1,
                                    getPhi(site, timeslice),
                                    coshold, sinhold
                                    );

                MatSmall emVNew = eVMatrix(
                                        -1,
                                        newphi,
                                        coshnew, sinhnew
                                        );

                this->Delta_i = emVNew * eVOld;
                this->Delta_i.diag() -= cpx(1.0,0.0);

            };

            void Simulation::calculateM(std::tuple<RowVecReal,int,int>& newlocalphi){
            // Compute "core matrix" M whose determinant equals the determinant ratio of the Green's functions
            // differing by a single local change. Also enters in iterative update of GF.

                int i;int l; RowVecReal newphi;
                std::tie(newphi,i, l) = newlocalphi;

                calculateDelta_i(newlocalphi);
                MatSmall G_i = ZeroSmall;
                G_i(0,0) = g(i,i);
                G_i(0,1) = g(i,i+simParams.N);
                G_i(1,0) = g(i+simParams.N,i);
                G_i(1,1) = g(i+simParams.N,i+simParams.N);

                this->M = EyeSmall+(EyeSmall - G_i)*this->Delta_i;
            };

            void Simulation::calculateMInverseAndDet(){
            // Compute inverse and determinant of "core matrix" M (2x2 matrix).

                this->Minverse = ZeroSmall;
                this->Mdet = M(0,0)*M(1,1)-M(0,1)*M(1,0);
                if(this->Mdet==cpx(0,0))
                    std::cout << "Inverting non-invertible matrix!" << std::endl;

                // Compute inverse
                this->Minverse(0,0) = M(1,1);
                this->Minverse(1,1) = M(0,0);
                this->Minverse(0,1) = (-1.0)*M(0,1);
                this->Minverse(1,0) = (-1.0)*M(1,0); 
                this->Minverse = (1.0/this->Mdet)*this->Minverse;

                /*std::cout << "calculateMInverseAndDet():" << std::endl;
                std::cout << "Minverse:" << this->Minverse << std::endl;
                std::cout << "det:" << this->Mdet << std::endl << std::endl;*/
            };


            cpx Simulation::calculateDetRatioForLocalUpdateNaive(CubeReal& phiField,int site, int timeslice){
            // Calculates the ratio of determinants of old and new fermionic Green's function as
            // necessary for Metropolis acceptance probability
            // Most direct implementation. Should only be used for consistency checks.
                using arma::det;

                Mat G_phi_tilde_l = calculateGFfromScratchNaive(this->g, timeslice);
                Mat delta = calculateDeltaNaive(timeslice, phiField);
                Mat arg = EyeSmall+delta*(EyeSmall - G_phi_tilde_l);
                return det(arg);
            };

            cpx Simulation::calculateDetRatioForLocalUpdateMax(std::tuple<RowVecReal,int,int>& newlocalphi){
            // Calculates the ratio of determinants of old and new fermionic Green's function, as
            // necessary for Metropolis acceptance probability, by using delta_i.
            //
            // Based on Max: updateInSlice_interative()

                calculateDelta_i(newlocalphi);

                int site = std::get<1>(newlocalphi);

                //****
                //Compute the determinant and inverse of I + Delta*(I - G)
                //based on Sherman-Morrison formula / Matrix-Determinant lemma
                //****

                std::array<Vec, MatrixSizeFactor> rows;

                //Delta*(I - G) is a sparse matrix containing just 4 rows:
                //site, site+N, site+2N, site+3N
                //Compute the values of these rows [O(N)]:
                for (uint32_t r = 0; r < MatrixSizeFactor; ++r) {
                    //TODO: Here are some unnecessary operations:
                    //Delta_i contains many repeated elements, and even
                    //some zeros
                    rows[r] = Vec(MatrixSizeFactor*simParams.N);
                    for (uint32_t col = 0; col < MatrixSizeFactor*simParams.N; ++col) {
                        rows[r][col] = -Delta_i(r,0) * g.col(col)[site];
                    }
                    rows[r][site] += Delta_i(r,0);
                    for (uint32_t dc = 1; dc < MatrixSizeFactor; ++dc) {
                        for (uint32_t col = 0; col < MatrixSizeFactor*simParams.N; ++col) { // Here stood a hardcoded 4 instead of MatrixSizeFactor
                            rows[r][col] += -Delta_i(r,dc) * g.col(col)[site + dc*simParams.N];
                        }
                        rows[r][site + dc*simParams.N] += Delta_i(r,dc);
                    }
                }

                // [I + Delta*(I - G)]^(-1) again is a sparse matrix
                // with two (four) rows site, site+N(, site+2N, site+3N).
                // compute them iteratively, together with the determinant of
                // I + Delta*(I - G)
                // Apart from these rows, the remaining diagonal entries of
                // [I + Delta*(I - G)]^(-1) are 1
                //
                // before this loop rows[] holds the entries of Delta*(I - G),
                // after the loop rows[] holds the corresponding rows of [I + Delta*(I - G)]^(-1)
                cpx det = 1;
                for (uint32_t l = 0; l < MatrixSizeFactor; ++l) {
                    Vec row = rows[l];
                    for (int k = int(l)-1; k >= 0; --k) {
                        row[site + unsigned(k)*simParams.N] = 0;
                    }
                    for (int k = int(l)-1; k >= 0; --k) {
                        row += rows[l][site + unsigned(k)*simParams.N] * rows[(unsigned)k];
                    }
                    cpx divisor = cpx(1) + row[site + l*simParams.N];
                    rows[l] = (-1.0/divisor) * row;
                    rows[l][site + l*simParams.N] += 1;
                    for (int k = int(l) - 1; k >= 0; --k) {
                        rows[(unsigned)k] -= (rows[unsigned(k)][site + l*simParams.N] / divisor) * row;
                    }
                    det *= divisor;
                }

                return det;
            };

            cpx Simulation::calculateDetRatioForLocalUpdate(std::tuple<RowVecReal,int,int>& newlocalphi){
            // Calculates the ratio of determinants of old and new fermionic Green's function, as
            // necessary for Metropolis acceptance probability, by using M, that is Delta_i and G_i.
            //
            // Consistent with calculateDetRatioForLocalUpdateMax()

                calculateM(newlocalphi);
                calculateMInverseAndDet();    

                // Comparing to calculateDetRatioForLocalUpdateMax
                /*cpx detratio2 = calculateDetRatioForLocalUpdateMax(newlocalphi);
                if(std::abs(this->Mdet-detratio2)>1.0e-12){
                    std::cout << "DetRatio: " << this->Mdet << ", Max: " << detratio2 << std::endl;
                    std::cout << "Diff: " << this->Mdet-detratio2 << std::endl << std::endl;
                }*/

                return this->Mdet;                      
            };

    // ## Metropolis criterium

        bool Simulation::acceptMetropolisLocal(std::tuple<RowVecReal,int,int>& newlocalphi){
        // Compare the present configuration to a new one and accept/reject the new one based on Metropolis
        // criterium
            // TODO: Tune random onsite phi generation such that ~50% of all proposals are being accepted.
            //       This probably affects most the scaling part of getRandomOnSitePhi?

            double DeltaS_phi = calculateBosonicActionChange(newlocalphi);
            double expactiondiff = exp(-DeltaS_phi);

            double p_acc=0;
            if(!simParams.noFermions){
                // Calculate acceptance probability in constant time
                cpx detratio = calculateDetRatioForLocalUpdate(newlocalphi);
                double detratioAbsSq = std::abs(detratio)*std::abs(detratio);

                p_acc = std::min(1.0,expactiondiff*detratioAbsSq);
            } else {
                p_acc = std::min(1.0,expactiondiff);
            }
                
            bool accept = this->randbool(p_acc);

            if (accept){
                // Update order parameter field
                setPhi(this->phiField,newlocalphi);
                this->S_phi = this->S_phi+DeltaS_phi;

                // Update GF
                if(!simParams.noFermions)
                    updateGFIterativelyInSlice(newlocalphi);

                ++this->logger.acceptedProposalsCount;
                return true;
            }
            return false;
        };

    // ## Measurements

        void Simulation::initMeasurements(){
        // Initialize all observables and set substage to MEASUREMENTS
            assert(stage==MonteCarlo::EQUILIBRIUM && "Initialization of measurements out of equilibrium!");

            occupationNumber = 0.0;
            meanPhi = arma::zeros<RowVecReal>(2);
            meanNormPhi = 0.0;
            SDWSuscept = 0.0;
            this->substage = MonteCarlo::MEASUREMENTS;
            measuring = true;
        };

        void Simulation::measure(int timeslice){
        // Measure observables at current sweep
            assert(substage==MonteCarlo::MEASUREMENTS && measuring && "Started measuring before initialization of measurements.");

            //measureOccupationNumber(occupationNumber, g, 1, simParams);
            measureMeanPhi(timeslice);
            measureMeanNormPhi(timeslice);
            ++simParams.M;
        };

        void Simulation::finishMeasurements(){
        // Finish measurements by finishing the calculation of averages 
            const auto N = simParams.N;
            const auto m = simParams.m;

            // for(int k=0; k<N; ++k){
            //     SDWSuscept += arma::dot(meanPhi,getPhi(k,0));
            // }
            // SDWSuscept /= N;
            SDWSuscept = arma::dot(meanPhi,getPhi(0,1));

            meanPhi /= (double)(N * m);
            normMeanPhi = arma::norm(meanPhi, 2);
            meanNormPhi /= (double)(N * m);

            meanPhiTimeSeries.push_back(meanPhi);
            normMeanPhiTimeSeries.push_back(normMeanPhi);
            meanNormPhiTimeSeries.push_back(meanNormPhi);
            SDWSusceptTimeSeries.push_back(SDWSuscept);
            times.push_back(simParams.currsweep);
            measuring = false;
        };

        void Simulation::measureMeanPhi(int timeslice){
            for (int site = 0; site < simParams.N; ++site) {
                RowVecReal phi = getPhi(site, timeslice);
                meanPhi += phi;
            }
        };

        void Simulation::measureMeanNormPhi(int timeslice){
            for (int site = 0; site < simParams.N; ++site) {
                RowVecReal phi = getPhi(site, timeslice);
                meanNormPhi += arma::norm(phi,2);
            }
        };

        void Simulation::measureOccupationNumber(){
        // Measures part of the mean occupation number, that is the occupation number averaged
        // over space-time lattice sites, spin and flavor.
        // Measures: Sum_i Sum_sigma Sum_alpha g_ii at timeslice of g
        // Complete Formula: n=1/mN Sum_i,l (1 - 1/D Sum_xi^D g_ii^\alpa\sigma)
            assert(!simParams.noFermions);

        // WARNING/TODO: We need to know g(l) for all l at this point (for the current configuration).
            auto getBlockOfGF = [&] (Mat& gf,int block) {
                // block = 1,2,3, or 4
                int blockdim = simParams.N;
                int row=(block<3)?0:1;
                int col=(block%2==0)?1:0;
                return gf.submat(row*blockdim,col*blockdim,(row+1)*blockdim-1,(col+1)*blockdim-1);
            };

            cpx gXupXupDiagonalSum = arma::sum(getBlockOfGF(g,1).diag());
            cpx gXupYdownDiagonalSum = arma::sum(getBlockOfGF(g,2).diag());
            cpx gYdownXupDiagonalSum = arma::sum(getBlockOfGF(g,3).diag());
            cpx gYdownYdownDiagonalSum = arma::sum(getBlockOfGF(g,4).diag());
            cpx totalsum = gXupXupDiagonalSum + gYdownYdownDiagonalSum + gXupYdownDiagonalSum + gYdownXupDiagonalSum;

            // We have to include the symmetry related part of the GF (Xdown Yup sector)
            // cpx gXdownXdownDiagonalSum = arma::conj(gXupXupDiagonalSum);
            // cpx gYupYupDiagonalSum = arma::conj(gYdownXdownDiagonalSum);
            // cpx gXdownYupDiagonalSum = arma::conj(gXupYdownDiagonalSum);
            // cpx gYupXdownDiagonalSum = arma::conj(gYdownXupDiagonalSum);
            // Adding those terms is equivalent to adding the conjugate of totalsum to totalsum.
            // Therefore we just take totalsum.real() and get the desired result.

            occupationNumber += totalsum.real();
        };

// ## Lattice/Model logic

    Cube Simulation::hoppingCube(){
    // Generates a hoppingCube, which stores the hopping matrix elements
    // row = spin (up, down), col = direction (horizontal, vertical), slices = flavor (x,y)
        Cube t = arma::zeros<Cube>(2,2,2);
        // So far: No magnetic field => no spin dependence 
        // OPT: Add magnetic field to avoid critical slowing down
        using arma::span;
        t(span::all,span(0),span(0)) = arma::cx_vec({modelParams.t.xh,modelParams.t.xh});
        t(span::all,span(0),span(1)) = arma::cx_vec({modelParams.t.yh,modelParams.t.yh});
        t(span::all,span(1),span(0)) = arma::cx_vec({modelParams.t.xv,modelParams.t.xv});
        t(span::all,span(1),span(1)) = arma::cx_vec({modelParams.t.yv,modelParams.t.yv});
        return t;
    };

    uVecInt Simulation::siteIndex2Coords(int site){
    // Receives linear site index (starting with 0) and
    // returns rowvec (row,col) specifying the coords of the site in 2D lattice
        const int L = simParams.L;
		unsigned int s = uint(site);
        uVecInt pos = {s/L,s-s/L*L}; // implicit rounding due to integer type
        return pos;
    };

    int Simulation::coords2SiteIndex(uVecInt pos){
    // Transforms site coordinates (uVecInt) to linear site index
        return pos(0)*simParams.L+pos(1);
    };

    int Simulation::coords2SiteIndex(uint x, uint y){
    // Transforms site coordinates (uVecInt) to linear site index
        return coords2SiteIndex(uVecInt({x,y}));
    };

    uVecInt Simulation::calculateNeighbors(int site){
    // Receive: linear site index
    // Calculate & Return: linear site indices of neighboring sites
    // Order: up (0), down (1), left (2), right (3)
        const int L = simParams.L;
        // Square lattice with periodic boundary conditions => 4 neighbors        
        int t = L*L;
        int row = site/L;
        uVecInt neighbors = arma::zeros<uVecInt>(4);
        // Switch 1 and 0 here to get back original left-handed coordinate system
        neighbors(1) = (site-L)<0?L*(L-1)+site:site-L;
        neighbors(0) = (site+L)<t?site+L:site-L*(L-1);
        neighbors(2) = (site-1)<(row*L)?site-1+L:site-1;
        neighbors(3) = (site+1)<(row*L+L)?site+1:site+1-L;
        return neighbors;
    };

    uVecInt Simulation::calculateTimeNeighbors(int timeslice){
    // Returns vector of time indices of neighboring timeslices
    // Entries: Previous = Earlier = 0, Subsequent = Later = 1
        uVecInt tnn = uVecInt(2);
        tnn(0) = (timeslice==1)?simParams.m:(timeslice-1);
        tnn(1) = (timeslice==simParams.m)?1:(timeslice+1);

        return tnn;
    };

    uVecInt Simulation::getNeighbors(int site){
    // Receive: linear site index
    // Return: linear site indices of neighboring sites
    // Order: up (0), down (1), left (2), right (3)
        return NeighborTable(arma::span::all,arma::span(site));
    };

    uVecInt Simulation::getTimeNeighbors(int timeslice){
    // Receive: time slice index
    // Return: time slice indices of neighboring time slices
    // Order: previous = Earlier (0), subsequent = Later (1)
        return TimeNeighborTable(arma::span::all,arma::span(timeslice));
    };

    uVecInt Simulation::getTopAndRightNeighbors(int site){
    // Receive: linear site index
    // Return: linear site indices of top and right neighboring site
    // 0 = up, 1 = right
        return NeighborTable(arma::uvec({0,3}),arma::uvec({uint(site)}));
    };

    void Simulation::createNeighborTable(){
    // Creates a lookup table of neighbors.
    // Columns correpond to different lattice sites.
    // The 4 rows contain site index of up (0), down (1), left (2) and right (3) neighbor.
        using arma::span;
        this->NeighborTable = uMatInt(4,simParams.N);
        for(int site=0;site<simParams.N;++site){
            NeighborTable(span::all,span(site)) = calculateNeighbors(site);
        }
    };

    void Simulation::createTimeNeighborTable(){
    // Creates a lookup table of time neighbors.
    // Columns correpond to different time slices.
    // The 2 rows contain time slice index of previous (0) and subsequent (1) time neighbor.
        using arma::span;
        this->TimeNeighborTable = uMatInt(2,simParams.m+1);
        for(int slice=0;slice<=simParams.m;++slice){
            TimeNeighborTable(span::all,span(slice)) = calculateTimeNeighbors(slice);
        }
    };


    // Checkerboard after Assaad for square lattice, i.e. two non-commuting kinetic exponentials (K_A and K_B)
    // representing groups (A and B) of commuting four-site hopping matrices 

    Mat Simulation::calculateKineticExponentialCheckerboard(int sign, double factor){
    // Calculates the matrix exponential e^(sign*factor*K_A)
        assert(simParams.L%2==0); // Assaad checkerboard (two hopping groups) only possible for even L
        assert( (sign==(-1)) || (sign==1) );
        int numPlaqEach = simParams.N/4;

        // First, identify sites that make up the L^2/4 hopping plaquettes
        // Herefore, we only identify the bottom left site. The others are up, right and up right neighbors.
        uVecInt sitesA = uVecInt(numPlaqEach);
        uVecInt sitesB = uVecInt(numPlaqEach);
        int idx=0;
        for(int x = 0; x<simParams.L; x=x+2){
            for(int y = 0; y<simParams.L; y=y+2){
                sitesA(idx) = coords2SiteIndex(x,y);
                sitesB(idx) = coords2SiteIndex(x+1,y+1);
                ++idx;
            }
        }

        // generic::print_var(sitesA);
        // generic::print_var(sitesB);

        uVecInt sitesAB = arma::join_cols(sitesA, sitesB);
        std::vector<SpMat> Kps; // First numPlaqEach are of group A, second half of group B

        for(int p=0; p<2*numPlaqEach; ++p){
            // Construct sparse matrix exponential of four-site hopping matrix KAp for each plaquette
            // sitesA(p) ^= ijkl in Max's TA,ijkl notation
            using std::cosh; using std::sinh;

            uVecInt kpos = siteIndex2Coords(sitesAB(p));
            int i = sitesAB(p);
            int j = getTopAndRightNeighbors(i)(1);
            int k = getTopAndRightNeighbors(i)(0);
            int l = getTopAndRightNeighbors(k)(1);

            SpMat expKp = arma::speye<SpMat>(2*simParams.N,2*simParams.N);
            // generic::block(expKp,MatrixSizeFactor,0,1) = arma::speye<SpMat>(simParams.N,simParams.N);
            // generic::block(expKp,MatrixSizeFactor,1,0) = arma::speye<SpMat>(simParams.N,simParams.N);
            expKp /= 2.0;
            Cube t = hoppingCube();
            
            // Sector 0=XUP 1=YDOWN (K is diagonal in full flavor+spin space.)
            for(int sector=0;sector<2;++sector) {
                // The 0.5 factor is for compensating the addition of the transpose below
                generic::block(expKp,MatrixSizeFactor,sector,sector)(i,i) = 0.5*cosh(factor*t(sector,0,sector))*cosh(factor*t(sector,1,sector));
                generic::block(expKp,MatrixSizeFactor,sector,sector)(j,j) = 0.5*cosh(factor*t(sector,0,sector))*cosh(factor*t(sector,1,sector));
                generic::block(expKp,MatrixSizeFactor,sector,sector)(k,k) = 0.5*cosh(factor*t(sector,0,sector))*cosh(factor*t(sector,1,sector));
                generic::block(expKp,MatrixSizeFactor,sector,sector)(l,l) = 0.5*cosh(factor*t(sector,0,sector))*cosh(factor*t(sector,1,sector));

                generic::block(expKp,MatrixSizeFactor,sector,sector)(i,j) = sign*(-1.0)*cosh(factor*t(sector,1,sector))*sinh(factor*t(sector,0,sector));
                generic::block(expKp,MatrixSizeFactor,sector,sector)(i,k) = sign*(-1.0)*cosh(factor*t(sector,0,sector))*sinh(factor*t(sector,1,sector));
                generic::block(expKp,MatrixSizeFactor,sector,sector)(i,l) = sinh(factor*t(sector,0,sector))*sinh(factor*t(sector,1,sector));
                generic::block(expKp,MatrixSizeFactor,sector,sector)(j,l) = sign*(-1.0)*cosh(factor*t(sector,0,sector))*sinh(factor*t(sector,1,sector));
                generic::block(expKp,MatrixSizeFactor,sector,sector)(j,k) = sinh(factor*t(sector,0,sector))*sinh(factor*t(sector,1,sector));
                generic::block(expKp,MatrixSizeFactor,sector,sector)(k,l) = sign*(-1.0)*cosh(factor*t(sector,1,sector))*sinh(factor*t(sector,0,sector));

                generic::block(expKp,MatrixSizeFactor,sector,sector) = generic::block(expKp,MatrixSizeFactor,sector,sector) * exp(sign*factor*(-modelParams.mu));

                generic::block(expKp,MatrixSizeFactor,sector,sector) += generic::block(expKp,MatrixSizeFactor,sector,sector).t();
            }

            Kps.push_back(expKp);

        };

        Mat result = Eye;

        for(SpMat Kp: Kps){
            result = Kp*result;
        }

        // int pos = std::find(sitesAB.begin(), sitesAB.end(), 7) - sitesAB.begin(); 
        // generic::print_var(pos);
        // Mat K8 = arma::eye<Mat>(simParams.N,simParams.N)*generic::block(Kps[pos],MatrixSizeFactor,0,0);
        // MatReal K8abs = arma::abs(K8);
        // generic::print_var(K8abs);

        return result;

    };


    // Mat Simulation::Btimes(Mat A, int l){

    // };

    // Mat Simulation::timesB(Mat A, int l){

    // };

// ## Random number generation

    double Simulation::randu(double min, double max){
    // Generates a random number drawn from uniform_real_distribution in range [min,max)
        boost::random::uniform_real_distribution<> dist(min,max);
        return dist(this->rng);        
    };
    double Simulation::randu01(){
    // Defines randu01(), which delivers a random number chosen from uniform_real_distribution in
    // range [0,1)
        return randu(0,1);        
    };
    int Simulation::randint(int min, int max){
    // Generates a random number drawn from uniform_int_distribution in range [min,max]
        boost::random::uniform_int_distribution<> dist(min,max);
        return dist(this->rng);        
    };
    bool Simulation::randbool(double prob_true) {
    // Generates a boolean which is true with probability prob_true
        // prob_true must be a probability 0<=prob_true<=1
        assert(0<=prob_true);
        assert(1>=prob_true);

        boost::random::bernoulli_distribution<> dist(prob_true);
        return dist(this->rng);
    };
    RowVecReal Simulation::getRandom2DVectorPolar(double max_length){
    // Generates a random two-dimensional row vector, given a max length constraint (polar represent.)
    // random numbers are generated by using randu of the rng within simulation sim
    // The resulting vector has the form vector = {radius, angle}, i.e. polar coordinates.
        RowVecReal vector(2);
        vector(0) = randu(0,max_length);
        vector(1) = randu(0,2.0*constants::pi);
        return vector;
    };
    RowVecReal Simulation::getRandom2DVector(double max_length){
    // Generates a random two-dimensional row vector, given a max length constraint
    // random numbers are generated by using randu of the rng within simulation sim
    // The resulting vector has the form vector = {x, y}, i.e. cartesian coordinates.
        return generic::polar2cartesian(getRandom2DVectorPolar(max_length));
    };
    RowVecReal Simulation::getRandom2DVectorBox(double box_length){
    // Generates a random two-dimensional row vector within a box of length box_length.acceptedProposalsCount
    // random numbers are generated by using randu of the rng within simulation sim
    // The resulting vector has the form vector = {x, y}, i.e. cartesian coordinates.
        return RowVecReal({randu(-box_length,box_length),randu(-box_length,box_length)});
    };

// ## Storage
    
    void Simulation::writeConfigurationToFile(std::string filename) {
    // Saves (only) the current configuration to file (c.f. writeStateToFile)
        Simulation::writeConfigurationToFile(this->phiField, filename);
    };
    void Simulation::writeStateToFile(std::string filename) {
    // Saves the current state of the simulation to file, i.e. present configuration, parameters etc.
        Simulation::writeStateToFile(*this, filename);
    };
    void Simulation::loadConfigurationFromFile(std::string filename) {
    // Loads a configuration file
        Simulation::loadConfigurationFromFile(filename, this->phiField);
    };
    void Simulation::loadStateFromFile(std::string filename) {
    // Load the state of a simulation from file, i.e. current configuration, parameters etc.
        Simulation::loadStateFromFile(filename, *this);
    };


    // (Static, generic versions)
    void Simulation::writeConfigurationToFile(CubeReal& phiField, std::string filename) {
    // Generic: Saves (only) the given configuration conf to file (c.f. writeStateToFile)
        phiField.save(filename,arma::arma_binary); //arma_binary, arma_ascii, raw_ascii
    };
    void Simulation::writeStateToFile(Simulation& sim, std::string filename) {
    // Generic: Saves the state of the simulation sim to file, i.e. present configuration, parameters etc.
        std::ofstream file;
        file.open(filename);
        file << "#?StateFile" << std::endl;
        file << "#Parameters" << std::endl;
        file << "#r,lambda,u,mu,txh,txv,tyh,tyv" << std::endl;
        file << sim.modelParams << std::endl;
        file << "#T,L,dtau,s,thermal sweeps,sweeps,current sweep,save rate, print rate, keepTempStates, DeltaPhiLength, adaptRounds, sweepsPerAdaptRound, adaptPercentage, currAcceptRateSum, ProposeMethod" << std::endl;
        file << sim.simParams << std::endl;
        file << "#proposalCount, acceptedProposalsCount" << std::endl;
        file << sim.logger << std::endl;
        file << "#Random generator state" << std::endl ;
        file << sim.rng << std::endl;
        file << "#Configuration" << std::endl;
        sim.phiField.save(file,arma::arma_binary);
        file.close();

        // Save GF to seperate file
        size_t lastindex = filename.find_last_of("."); 
        std::string rawname = filename.substr(0, lastindex); 
        sim.g.save(rawname+std::string(".gf"),arma::arma_binary);
    };
    void Simulation::loadConfigurationFromFile(std::string filename, CubeReal& phiField) {
    // Generic: Loads system configuration conf from a configuration file
        phiField.load(filename,arma::arma_binary); //arma_binary, arma_ascii, raw_ascii
    };
    void Simulation::loadStateFromFile(std::string filename, Simulation& sim) {
    // Generic:: Loads the state of a simulation sim from file, i.e. current configuration, parameters etc.
    // TODO: Make more tolerant, e.g. to small variations in state files.
        std::ifstream file;
        file.open(filename);

        // Parsing
        std::string line;
        std::getline(file, line);
        if (line!="#?StateFile") {
            throw StorageException("Not a proper state file.");
        }
        std::getline(file, line); // comment line
        std::getline(file, line); // comment line
        
        std::getline(file, line); // ModelParameters line
        unsigned first = line.find("(");
        unsigned last = line.find(")");
        std::string args = line.substr(first+1,last-first-1);
        std::string d;
        std::istringstream iss(args);
        std::getline(iss,d,',');
        sim.modelParams.r = generic::str2double(d);
        std::getline(iss,d,',');
        sim.modelParams.lambda = generic::str2double(d);
        std::getline(iss,d,',');
        sim.modelParams.u = generic::str2double(d);
        std::getline(iss,d,',');
        sim.modelParams.mu = generic::str2double(d);
        std::getline(iss,d,',');
        sim.modelParams.t.xh = generic::str2double(d);
        std::getline(iss,d,',');
        sim.modelParams.t.xv = generic::str2double(d);
        std::getline(iss,d,',');
        sim.modelParams.t.yh = generic::str2double(d);
        std::getline(iss,d,',');
        sim.modelParams.t.yv = generic::str2double(d);

        std::getline(file, line); // comment line

        std::getline(file, line); // SimulationParameters line
        first = line.find("(");
        last = line.find(")");
        args = line.substr(first+1,last-first-1);
        std::istringstream isss(args);
        std::getline(isss,d,',');
        sim.simParams.T = generic::str2double(d);
        std::getline(isss,d,',');
        sim.simParams.L = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.dtau = generic::str2double(d);
        std::getline(isss,d,',');
        sim.simParams.s = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.thermalSweeps = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.sweeps = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.currsweep = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.saveRate = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.printRate = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.keepTempStates = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.DeltaPhiLength = generic::str2double(d);
        std::getline(isss,d,',');
        sim.simParams.adaptRounds = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.sweepsPerAdaptRound = generic::str2int(d);
        std::getline(isss,d,',');
        sim.simParams.adaptPercentage = generic::str2double(d);
        std::getline(isss,d,',');
        sim.simParams.currAcceptRateSum = generic::str2double(d);
        std::getline(isss,d,',');
        sim.simParams.proposeMethod = ((d=="CIRCLE")?SimulationParameters::CIRCLEPROPOSE:SimulationParameters::BOXPROPOSE);

        std::getline(file, line); // comment line

        std::getline(file, line); // Logger line
        first = line.find("(");
        last = line.find(")");
        args = line.substr(first+1,last-first-1);
        std::istringstream issss(args);
        std::getline(issss,d,',');
        sim.logger.proposalCount = generic::str2int(d);
        std::getline(issss,d,',');
        sim.logger.acceptedProposalsCount = generic::str2int(d);
        std::getline(issss,d,',');
        sim.logger.proposalCountDuringAdaption = generic::str2int(d);
        std::getline(issss,d,',');
        sim.logger.acceptedProposalsCountDuringAdaption = generic::str2int(d);

        std::getline(file, line); // comment line
        std::getline(file, line); // get rng state
        std::istringstream rngstream(line);
        rngstream >> sim.rng; 

        std::getline(file, line); // comment line
        //std::cout << line << std::endl;
        sim.phiField.load(file,arma::arma_binary); // load system configuration (CubeReal)

        file.close();

        // Load GF from seperate file
        size_t lastindex = filename.find_last_of("."); 
        std::string rawname = filename.substr(0, lastindex); 
        sim.g.load(rawname+std::string(".gf"),arma::arma_binary); // load fermion GF (Mat)

        // Init simulation with restored settings
        sim.initResume();
    };

    void Simulation::timeSeriesToFile(std::vector<int> times, std::vector<RowVecReal> timeSeries, std::string filename){
        assert(times.size()==timeSeries.size());

        std::ofstream file;
        file.open(filename);
        for(int t=0; t<times.size(); ++t)
            file << times.at(t) << "\t" << timeSeries.at(t)(0) << "\t" << timeSeries.at(t)(1) << std::endl;
        file.close();
    };

    void Simulation::timeSeriesToFile(std::vector<int> times, std::vector<double> timeSeries, std::string filename){
        assert(times.size()==timeSeries.size());

        std::ofstream file;
        file.open(filename);
        for(int t=0; t<times.size(); ++t)
            file << times.at(t) << "\t" << generic::num2str(timeSeries.at(t)) << std::endl;
        file.close();
    };

#endif
