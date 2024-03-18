/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
 *
 * This file is part of DOpElib
 *
 * DOpElib is free software: you can redistribute it
 * and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version.
 *
 * DOpElib is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * Please refer to the file LICENSE.TXT included in this distribution
 * for further information on this license.
 *
 **/

//c++ includes
#include <iostream>
#include <fstream>

//deal.ii includes
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#if DEAL_II_VERSION_GTE(9,1,1)
#else
#include <deal.II/grid/tria_boundary_lib.h>
#endif
#include <deal.II/grid/grid_generator.h>

//DOpE includes
#include <include/parameterreader.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <basic/mol_spacetimehandler.h> 
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>
#include <templates/newtonsolver.h>
#include <interfaces/functionalinterface.h>

#include <problemdata/noconstraints.h> 

//Changed - mixed includes. 
#include <templates/integratormixeddims.h> // for mixed dim opt. control
#include <templates/newtonsolvermixeddims.h> // for mixed dim opt. control

//DOpE includes for instationary problems
#include <reducedproblems/instatpdeproblem.h>
#include <container/instatpdeproblemcontainer.h>
#include <container/dwrdatacontainer.h>
// Changed
#include <reducedproblems/instatreducedproblem.h>
#include <templates/instat_step_newtonsolver.h>
#include <opt_algorithms/reducednewtonalgorithm.h>
#include <container/instatoptproblemcontainer.h>

// Timestepping scheme
#include <tsschemes/backward_euler_problem.h>
#include <tsschemes/crank_nicolson_problem.h>
#include <tsschemes/shifted_crank_nicolson_problem.h>
#include <tsschemes/fractional_step_theta_problem.h>

// A new heuristic Newton solver
#include "instat_step_modified_newtonsolver.h"

// PDE
#include "localpde.h"


// Finally, as in the other DOpE examples, we
// have goal functional evaluations and problem-specific data.
#include "functionals.h"
#include "problem_data.h"
#include "localfunctional.h"

//For Loading Data
#include "VectorWriteLoad.cc"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define dimensions for control- and state problem
const static int DIM = 2;
const static int CDIM = 0; 

#if DEAL_II_VERSION_GTE(9,3,0)
#define DOFHANDLER false
#else
#define DOFHANDLER DoFHandler
#endif
#define FE FESystem
#define EDC ElementDataContainer
#define FDC FaceDataContainer


/*********************************************************************************/
//Use LobattoFormulas, as obstacle multiplier is located in vertices
typedef QGaussLobatto<DIM> QUADRATURE;
typedef QGaussLobatto<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

typedef FunctionalInterface<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNC;

#define TSP BackwardEulerProblem
#define DTSP BackwardEulerProblem

typedef InstatOptProblemContainer<TSP, DTSP, FUNC,
        LocalFunctional<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>,
        LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        NoConstraints<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE,
        FACEQUADRATURE, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef IntegratorMixedDimensions<IDC, VECTOR, double, CDIM, DIM> INTEGRATORM; //This Integrator added due to mixed dimension stuff

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef VoidLinearSolver<VECTOR> VOIDLS;//dummy solver for the 0d control

typedef InstatStepModifiedNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef NewtonSolverMixedDimensions<INTEGRATORM, VOIDLS, VECTOR> CNLS; //for mixed dimension stuff
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATORM, INTEGRATOR, OP, VECTOR, CDIM,
        DIM> RP;


void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("prerefine", "5", Patterns::Integer(1),
                             "How often should we refine the coarse grid?");
  param_reader.declare_entry("num_intervals", "75", Patterns::Integer(1),
                               "How many quasi-timesteps?");
}

/*********************************************************************************/

int
main(int argc, char **argv)
{
  /**
   *  We solve a quasi-static phase-field brittle fracture
   *  propagation problem. The crack irreversibility
   *  constraint is imposed with the help of a Lagrange multiplier
   *  The configuration is the single edge notched tension test.
   */

  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv);
  
  string paramfile = "dope.prm";

  if (argc == 2)
    {
      paramfile = argv[1];
    }
  else if (argc > 2)
    {
      std::cout << "Usage: " << argv[0] << " [ paramfile ] " << std::endl;
      return -1;
    }

  /*********************************************************************************/
  // Parameter data
  ParameterReader pr;
  declare_params(pr);
  RP::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  LocalFunctional<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>::declare_params(pr);
  NonHomoDirichletData::declare_params(pr);
  //LocalBoundaryFunctionalStressX<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>::declare_params(pr);
  //LocalFunctionalBulk<EDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  //LocalFunctionalCrack<EDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  RNA::declare_params(pr); 
  pr.read_parameters(paramfile);
  
  /*********************************************************************************/
  // Reading mesh and creating triangulation
  pr.SetSubsection("main parameters");
  int prerefine = pr.get_integer("prerefine");
  int num_intervals = pr.get_integer("num_intervals");
  
  Triangulation<DIM> triangulation;
  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("unit_slit.inp");
  grid_in.read_ucd(input_file);
  triangulation.refine_global(prerefine);
  
  /*********************************************************************************/
  //Create Control FE
  FE<DIM> control_fe(FE_Nothing<DIM>(1), 1);

  /*********************************************************************************/
  // Assigning finite elements
  // (1) = polynomial degree - please test (2) for u and phi
  // zweite Zahl: Anzahl der Komponenten
  FE<DIM> state_fe(FE_Q<DIM>(2), 2, // vector-valued (here dim=2): displacements 
		   FE_Q<DIM>(1), 1, // scalar-valued phase-field
    		   FE_Q<DIM>(1), 1);  // scalar-valued multiplier for irreversibility

  QUADRATURE quadrature_formula(3);
  FACEQUADRATURE face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  /*********************************************************************************/
  // Defining the specific PDE
  LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);

  /**********************************************************************/
  /*****
  // Defining goal functional
  LocalBoundaryFunctionalStressX<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LBFSX(pr);
  LocalFunctionalBulk<EDC, FDC, DOFHANDLER, VECTOR, DIM> LFB(pr);
  LocalFunctionalCrack<EDC, FDC, DOFHANDLER, VECTOR, DIM> LFC(pr);
  *****/
  
  LocalFunctional<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LFunc(pr); //Cost functional

  /*********************************************************************************/
  // Create a time grid of [0,0.0075] with 75 timeintervals
  Triangulation<1> times;
  double initial_time = 0.0;
  double end_time = 0.0075;
  GridGenerator::subdivided_hyper_cube(times, num_intervals, initial_time, end_time);

  /*********************************************************************************/
  // We give the spatial and time triangulation as well as the state finite
  // elements to the MOL-space time handler.
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, CDIM, DIM>
  DOFH(triangulation, control_fe, state_fe, times, DOpEtypes::VectorAction::stationary);

  NoConstraints<EDC, FDC, DOFHANDLER, VECTOR, CDIM,
                DIM> Constraints; // Mostly as dummy for now, might need to put actual restraints here later
  // Finales Problem, was alles handled
  OP P(LFunc, LPDE, Constraints, DOFH);
  /*********************************************************************************/
  //Load Reference solution, the distance to which should be minimized. (Specifically the distance to displacement and phase field)
  //The reference files contain the vectors in plain text, simply listing all entries in order.
  //An additional file is used to remember the number of blocks in each vector and the size of these blocks.
  StateVector<VECTOR> statesolution(&DOFH, DOpEtypes::VectorStorageType::fullmem, pr);
  vector<double> BlockSize;
  vector<double> VectorContent;
  unsigned int count;
  for(unsigned int j = 0; j <= 75; j++)
  {
  BlockSize.clear();
  VectorContent.clear();
  readVectorFromFile("ReferenceData/VectorContent"+to_string(j)+".txt", VectorContent);
  readVectorFromFile("ReferenceData/VectorStructure"+to_string(j)+".txt", BlockSize);
  BlockVector<double> stateVector;
  vector<unsigned int> block_sizes;
  for(unsigned int i = 0; i < BlockSize.size(); ++i)
  {
   block_sizes.push_back(static_cast<unsigned int>(BlockSize[i]));
  }
  stateVector.reinit(block_sizes);
  count = 0;
  for(unsigned int i = 0; i < BlockSize.size(); ++i)
  {
   for(unsigned int k = 0; k < block_sizes[i]; ++k)
   {
    stateVector.block(i)(k) = VectorContent[count];
    count++;
   }
  }
  statesolution.SetTimeDoFNumber(j);
  statesolution.GetSpacialVector() = stateVector;
  }
  P.AddAuxiliaryState(&statesolution, "ReferenceSolution");

  /*********************************************************************************/
  /*****
  // Add quantity of interest to the problem
  P.AddFunctional(&LBFSX);
  P.AddFunctional(&LFB);
  P.AddFunctional(&LFC);
  

  // We want to evaluate the stress on the top boundary with id = 3
  P.SetBoundaryFunctionalColors(3);
  ******/


  /*********************************************************************************/
  // Prescribing boundary values
  // We have 3 components (2D displacements and scalar-valued phase-field)
  // 4 components with u(x), u(y), phi(x), tau(x)
  std::vector<bool> comp_mask(4);
  comp_mask[2] = false; // phase-field component (always hom. Neumann data)
   
  // Fixed boundaries
  DOpEWrapper::ZeroFunction<DIM> zf(4);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  // Non-homogeneous boudary (on top where we tear)
  NonHomoDirichletData dirichlet_data(pr);
  SimpleDirichletData<VECTOR, DIM> DD2(dirichlet_data);

  // Bottom boundary -> TODO: Isn't this the top boundary? Probably comments just switched in the original?
  comp_mask[0] = true;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(2, comp_mask, &DD2);

  // Top boundary (shear force via non-homo. Dirichlet)
  comp_mask[0] = false;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(3, comp_mask, &DD1);

  /*********************************************************************************/
  // Initial data
  InitialData initial_data;
  P.SetInitialValues(&initial_data);

  /*********************************************************************************/
  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  /*******
  //Only needed for pure PDE Problems: We define and register
  //the output- and exception handler. The first handels the
  //output on the screen as well as the output of files. The
  //amount of the output can be steered by the paramfile.
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);
  ******/
  
  //Changed
  RNA Alg(&P, &solver, pr);
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem,pr);
  //q.SetTimeDoFNumber(0);
  //q.GetSpacialVector().reinit(1,1);
  //q.GetSpacialVector().block(0)(0) = 2.0;
  Alg.ReInit();
  
  //**************************************************************************************************
 
  try
    {
    /*******
      //Before solving we have to reinitialize the stateproblem and outputhandler.
      solver.ReInit();
      out.ReInit();

      stringstream outp;
      outp << "**************************************************\n";
      outp << "*             Starting Forward Solve             *\n";
      outp << "*   Solving : " << P.GetName() << "\t*\n";
      outp << "*   SDoFs   : ";
      solver.StateSizeInfo(outp);
      outp << "**************************************************";
      //We print this header with priority 1 and 1 empty line in front and after.
      out.Write(outp, 1, 1, 1);

      //We compute the value of the functionals. To this end, we have to solve
      //the PDE at hand.
      solver.ComputeReducedFunctionals();
      *****/
      Alg.Solve(q);

    }
  catch (DOpEException &e)
    {
      std::cout
          << "Warning: During execution of `" + e.GetThrowingInstance()
          + "` the following Problem occurred!" << std::endl;
      std::cout << e.GetErrorMessage() << std::endl;
    }

  return 0;
}

#undef FDC
#undef EDC
#undef FE
#undef DOFHANDLER

