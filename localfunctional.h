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

#ifndef LOCALFunctional_
#define LOCALFunctional_

//#include <interfaces/pdeinterface.h>
#include <interfaces/functionalinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dopedim, int dealdim>
  class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
#endif
{
public:

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("constant_k", "0.0", Patterns::Double(0));
    param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
    param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));
    param_reader.declare_entry("sigma", "1", Patterns::Double(0),
				"Which sigma in complementarity function");
  }
  
  LocalFunctional()
  {
  }
  
  LocalFunctional(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    alpha_eps_ = param_reader.get_double("alpha_eps");
    lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");
    s_ = param_reader.get_double("sigma");
  }
  
  

  bool
  NeedTime() const
  {
    return true;
  }

  double
  ElementValue(
    const EDC<DH, VECTOR, dealdim> &edc)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();

    uvalues_.resize(n_q_points, Vector<double>(4));
    edc.GetValuesState("state", uvalues_);

    refvalues_.resize(n_q_points, Vector<double>(4));
    edc.GetValuesState("ReferenceSolution", refvalues_);

    double ret = 0;
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        ret += 0.5 * ( (uvalues_[q_point][0] - refvalues_[q_point][0]) * (uvalues_[q_point][0] - refvalues_[q_point][0])
              + (uvalues_[q_point][1] - refvalues_[q_point][1]) * (uvalues_[q_point][1] - refvalues_[q_point][1])
              + (uvalues_[q_point][2] - refvalues_[q_point][2]) * (uvalues_[q_point][2] - refvalues_[q_point][2]) )
              * state_fe_values.JxW(q_point);
      }
    return ret;
  }

  void
  ElementValue_U(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    uvalues_.resize(n_q_points, Vector<double>(4));
    edc.GetValuesState("state", uvalues_);

    refvalues_.resize(n_q_points, Vector<double>(4));
    edc.GetValuesState("ReferenceSolution", refvalues_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar phasefield(2);
    Tensor<1, 2> phi_u;
    double phi_pf;

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        phi_u = state_fe_values[displacements].value(i, q_point);
        phi_pf = state_fe_values[phasefield].value(i, q_point);
        local_vector(i) += scale * ( (uvalues_[q_point][0] - refvalues_[q_point][0]) * phi_u[0]
                                   + (uvalues_[q_point][1] - refvalues_[q_point][1]) * phi_u[1]
                                   + (uvalues_[q_point][2] - refvalues_[q_point][2]) * phi_pf )
                           * state_fe_values.JxW(q_point);
      }
    }
  }

  void
  ElementValue_Q(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &, double)
  {
  }

  void
  ElementValue_UU(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    duvalues_.resize(n_q_points, Vector<double>(4));
    edc.GetValuesState("tangent", duvalues_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar phasefield(2);
    Tensor<1, 2> phi_u;
    double phi_pf;
  
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
       {
         for (unsigned int i = 0; i < n_dofs_per_element; i++)
           {
              phi_u = state_fe_values[displacements].value(i, q_point);
              phi_pf = state_fe_values[phasefield].value(i, q_point);
              local_vector(i) += scale *( duvalues_[q_point][0] * phi_u[0]
                                   + duvalues_[q_point][1] * phi_u[1]
                                   + duvalues_[q_point][2] * phi_pf )
                                 * state_fe_values.JxW(q_point);
           }
       }
  }

  void
  ElementValue_QU(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &, double)
  {
  }

  void
  ElementValue_UQ(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &, double)
  {
  }

  void
  ElementValue_QQ(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &, double)
  {
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_quadrature_points;
  }

  string
  GetType() const
  {
    return "domain timedistributed";
  }

  std::string
  GetName() const
  {
    return "Cost-functional";
  }

private:
  vector<Vector<double> > uvalues_;
  vector<Vector<double> > refvalues_;
  vector<Vector<double> > duvalues_;
  
  double constant_k_, alpha_eps_, lame_coefficient_mu_, lame_coefficient_lambda_, s_;
};
#endif
