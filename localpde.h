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

#ifndef LOCALPDE_
#define LOCALPDE_

#include <interfaces/pdeinterface.h>

#include "stress_splitting.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9, 3, 0)
template <
    template <bool DH, typename VECTOR, int dealdim> class EDC,
    template <bool DH, typename VECTOR, int dealdim> class FDC,
    bool DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#else
template <
    template <template <int, int> class DH, typename VECTOR, int dealdim> class EDC,
    template <template <int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template <int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
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

  LocalPDE(ParameterReader &param_reader) : state_block_component_(4, 0)
  {
    state_block_component_[1] = 1;
    state_block_component_[2] = 2;
    state_block_component_[3] = 3;

    param_reader.SetSubsection("Local PDE parameters");

    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    alpha_eps_ = param_reader.get_double("alpha_eps");
    lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");

    s_ = param_reader.get_double("sigma");
  }


  // Domain values for elements
  void
  ElementEquation(
      const EDC<DH, VECTOR, dealdim> &edc,
      dealii::Vector<double> &local_vector, double scale,
      double /*scale_ico*/)
  {
    // Copied from example. Will modify almost only G_c - Look at how edc works
    assert(this->problem_type_ == "state");
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
        edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
    last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    edc.GetValuesState("last_time_solution", last_timestep_uvalues_);

    // changed
    qvalues_.reinit(1);
    edc.GetParamValues("control", qvalues_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar phasefield(2);
    const FEValuesExtractors::Scalar multiplier(3);

    Tensor<2, 2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

    Tensor<2, 2> zero_matrix;
    zero_matrix.clear();

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      Tensor<2, 2> grad_u;
      grad_u.clear();
      grad_u[0][0] = ugrads_[q_point][0][0];
      grad_u[0][1] = ugrads_[q_point][0][1];
      grad_u[1][0] = ugrads_[q_point][1][0];
      grad_u[1][1] = ugrads_[q_point][1][1];

      Tensor<1, 2> u;
      u.clear();
      u[0] = uvalues_[q_point](0);
      u[1] = uvalues_[q_point](1);

      Tensor<1, 2> grad_pf;
      grad_pf.clear();
      grad_pf[0] = ugrads_[q_point][2][0];
      grad_pf[1] = ugrads_[q_point][2][1];

      double pf = uvalues_[q_point](2);
      double old_timestep_pf = last_timestep_uvalues_[q_point](2);

      const Tensor<2, 2> E = 0.5 * (grad_u + transpose(grad_u));
      const double tr_E = trace(E);

      Tensor<2, 2> stress_term;
      stress_term.clear();
      stress_term = lame_coefficient_lambda_ * tr_E * Identity + 2 * lame_coefficient_mu_ * E;

      Tensor<2, 2> stress_term_plus;
      Tensor<2, 2> stress_term_minus;

      // Necessary because stress splitting does not work
      // in the very initial time step.
      if (this->GetTime() > 0.001)
      {
        decompose_stress(stress_term_plus, stress_term_minus,
                         E, tr_E, zero_matrix, 0.0,
                         lame_coefficient_lambda_,
                         lame_coefficient_mu_, false);
      }
      else
      {
        stress_term_plus = stress_term;
        stress_term_minus = 0;
      }

      for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        // const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(i,q_point);
        const Tensor<2, 2> phi_i_grads_u = state_fe_values[displacements].gradient(i, q_point);
        const double phi_i_pf = state_fe_values[phasefield].value(i, q_point);
        const Tensor<1, 2> phi_i_grads_pf = state_fe_values[phasefield].gradient(i, q_point);

        // Solid (Time-lagged version)
        local_vector(i) += scale * (scalar_product(((1.0 - constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) * stress_term_plus, phi_i_grads_u) + scalar_product(stress_term_minus, phi_i_grads_u)) * state_fe_values.JxW(q_point);

        // Phase-field
        local_vector(i) += scale * (
                                       // Main terms
                                       (1.0 - constant_k_) * scalar_product(stress_term_plus, E) * pf * phi_i_pf - qvalues_[0] / (alpha_eps_) * (1.0 - pf) * phi_i_pf + qvalues_[0] * alpha_eps_ * grad_pf * phi_i_grads_pf) *
                           state_fe_values.JxW(q_point);

        // Now the inequality constraint.
        // Evaluate only in vertices, so we check whether the lambda test function
        //  is one (i.e. we are in a vertex)

        if (fabs(state_fe_values[multiplier].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon())
        {
          // Weight to account for multiplicity when running over multiple meshes.
          unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
          double weight = 1. / n_neig;
          if (n_neig == 4)
          {
            // Equation for multiplier
            local_vector(i) += scale * weight * (uvalues_[q_point][3] - std::max(0., uvalues_[q_point][3] + s_ * (pf - old_timestep_pf)));
            // Add Multiplier to the state equation
            // find corresponding basis of state
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {
              if (fabs(state_fe_values[phasefield].value(j, q_point) - 1.) < std::numeric_limits<double>::epsilon())
              {
                local_vector(j) += scale * weight * uvalues_[q_point][3]; // TODO: This probably the last term of second row?
              }
            }
          }
          else // Boundary or hanging node (no weight, so it works if hanging)
          {
            local_vector(i) += scale * uvalues_[q_point][3];
          }
        }
      }
    }
    
  }
  // Domain values for elements
  void
  ElementEquation_U(
      const EDC<DH, VECTOR, dealdim> &edc,
      dealii::Vector<double> &local_vector, double scale,
      double /*scale_ico*/)
  {
      assert(this->problem_type_ == "adjoint");
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
  unsigned int n_q_points = edc.GetNQPoints();

  const FEValuesExtractors::Vector displacements(0);
  const FEValuesExtractors::Scalar phasefield(2);
  const FEValuesExtractors::Scalar multiplier(3);

  uvalues_.resize(n_q_points, Vector<double>(4));
  ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));

  edc.GetValuesState("state", uvalues_);
  edc.GetGradsState("state", ugrads_);

  zvalues_.resize(n_q_points, Vector<double>(4));
  zgrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("last_newton_solution", zvalues_);
  edc.GetGradsState("last_newton_solution", zgrads_);

  edc.GetValuesState("last_time_solution", last_timestep_uvalues_);

  // changed
  qvalues_.reinit(1);
  edc.GetParamValues("control", qvalues_);

  std::vector<Tensor<2, 2>> phi_grads_u(n_dofs_per_element);
  std::vector<double> phi_pf(n_dofs_per_element);
  std::vector<Tensor<1, 2>> phi_grads_pf(n_dofs_per_element);

  Tensor<2, 2> Identity;
  Identity[0][0] = 1.0;
  Identity[1][1] = 1.0;

  Tensor<2, 2> zero_matrix;
  zero_matrix.clear();

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    for (unsigned int k = 0; k < n_dofs_per_element; k++)
    {
      phi_grads_u[k] = state_fe_values[displacements].gradient(k, q_point);
      phi_pf[k] = state_fe_values[phasefield].value(k, q_point);
      phi_grads_pf[k] = state_fe_values[phasefield].gradient(k, q_point);
    }

    Tensor<2, 2> grad_u;
    grad_u.clear();
    grad_u[0][0] = ugrads_[q_point][0][0];
    grad_u[0][1] = ugrads_[q_point][0][1];
    grad_u[1][0] = ugrads_[q_point][1][0];
    grad_u[1][1] = ugrads_[q_point][1][1];

    Tensor<1, 2> v;
    v[0] = uvalues_[q_point](0);
    v[1] = uvalues_[q_point](1);

    Tensor<1, 2> grad_pf;
    grad_pf.clear();
    grad_pf[0] = ugrads_[q_point][2][0];
    grad_pf[1] = ugrads_[q_point][2][1];

    double pf = uvalues_[q_point](2);
    double old_timestep_pf = last_timestep_uvalues_[q_point](2);

    const Tensor<2, 2> E = 0.5 * (grad_u + transpose(grad_u));
    const double tr_E = trace(E);

    Tensor<2, 2> stress_term;
    stress_term.clear();
    stress_term = lame_coefficient_lambda_ * tr_E * Identity + 2 * lame_coefficient_mu_ * E;

    Tensor<2, 2> stress_term_plus;
    Tensor<2, 2> stress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(stress_term_plus, stress_term_minus,
                       E, tr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, false); // false as u terms only appear in sigma, not in deriv sigma
    }
    else
    {
      stress_term_plus = stress_term;
      stress_term_minus = 0;
    }

    // Prepare zgrads for use by splitting it into its components

    // First the displacement part of z
    Tensor<2, 2> grad_zDisp;
    grad_zDisp.clear();
    grad_zDisp[0][0] = zgrads_[q_point][0][0];
    grad_zDisp[0][1] = zgrads_[q_point][0][1];
    grad_zDisp[1][0] = zgrads_[q_point][1][0];
    grad_zDisp[1][1] = zgrads_[q_point][1][1];
    // Next phase field part
    Tensor<1, 2> grad_zPf;
    grad_zPf.clear();
    grad_zPf[0] = zgrads_[q_point][2][0];
    grad_zPf[1] = zgrads_[q_point][2][1];
    // Lastly gradients of z coresponding to the multiplier
    Tensor<1, 2> grad_zMult;
    grad_zMult.clear();
    grad_zMult[0] = zgrads_[q_point][3][0];
    grad_zMult[1] = zgrads_[q_point][3][1];

    // prepare zvalues as well
    Tensor<2, 1> zDisp;
    zDisp[0] = zvalues_[q_point][0];
    zDisp[1] = zvalues_[q_point][1];
    double zPf = zvalues_[q_point][2];
    double zMult = zvalues_[q_point][3];

    // Now calculate E_lin and sigma + and - for z
    const Tensor<2, 2> zE = 0.5 * (grad_zDisp + transpose(grad_zDisp));
    const double ztr_E = trace(zE);

    Tensor<2, 2> zstress_term;
    zstress_term.clear();
    zstress_term = lame_coefficient_lambda_ * ztr_E * Identity + 2 * lame_coefficient_mu_ * zE;

    Tensor<2, 2> zstress_term_plus;
    Tensor<2, 2> zstress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(zstress_term_plus, zstress_term_minus,
                       zE, ztr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, true); // true as z terms only appear in deriv sigma not in sigma itself
    }
    else
    {
      zstress_term_plus = stress_term;
      zstress_term_minus = 0;
    }

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
    {
      // Solid (time-lagged version)
      local_vector(i) += scale * (scalar_product(((1 - constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) * zstress_term_plus, phi_grads_u[i]) // du
                                  + scalar_product(zstress_term_minus, phi_grads_u[i])                                                                      // du
                                  ) *
                         state_fe_values.JxW(q_point);

      // Phase-field
      local_vector(i) += scale * (
                                     // Main terms
                                     (1 - constant_k_) * (scalar_product(zstress_term_plus, E) + scalar_product(stress_term_plus, zE)) * pf * phi_pf[i] // du
                                     + (1 - constant_k_) * scalar_product(stress_term_plus, E) * phi_pf[i] * zPf                                        // d phi
                                     + qvalues_[0] / (alpha_eps_)*phi_pf[i] * zPf                                                                       // d phi
                                     + qvalues_[0] * alpha_eps_ * phi_grads_pf[i] * grad_zPf                                                            // d phi
                                     ) *
                         state_fe_values.JxW(q_point);

      // Now the Multiplierpart
      // only in vertices, so we check whether one of the
      // lambda test function
      //  is one (i.e. we are in a vertex)
      if (
          (fabs(state_fe_values[multiplier].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon()))
      {
        // Weight to account for multiplicity when running over multiple meshes.
        unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
        double weight = 1. / n_neig;

        if (n_neig == 4)
        {
          // max = 0
          if ((uvalues_[q_point][3] + s_ * (pf - old_timestep_pf)) <= 0.)
          {
            local_vector(i) += scale * weight * state_fe_values[multiplier].value(i, q_point) * zMult;
          }
          else // max > 0
          {
            // From Complementarity
            local_vector(i) -= scale * weight * s_ * zPf * state_fe_values[multiplier].value(i, q_point);
          }
          // From Equation
          local_vector(i) += scale * weight * state_fe_values[phasefield].value(i, q_point) * zMult;
        }
        else // Boundary or hanging node no weight so it works when hanging
        {
          local_vector(i) += scale * state_fe_values[multiplier].value(i, q_point) * zMult; // non-derivative is qvalue, but that is actually <Lambda, multFE_i>, but letter is 1 in this bracket and hence does not show up in elementequation. Derive to lambda and seting in multFE_j gives element matrix, so this should be <zmult, multFE_i>
        }
      }
    }
  }

}
// Domain values for elements
void
ElementEquation_UT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
{
  assert(this->problem_type_ == "tangent");
  const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
  unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
  unsigned int n_q_points = edc.GetNQPoints();

  const FEValuesExtractors::Vector displacements(0);
  const FEValuesExtractors::Scalar phasefield(2);
  const FEValuesExtractors::Scalar multiplier(3);

  uvalues_.resize(n_q_points, Vector<double>(4));
  ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));

  edc.GetValuesState("last_newton_solution", uvalues_);
  edc.GetGradsState("last_newton_solution", ugrads_);

  duvalues_.resize(n_q_points, Vector<double>(4));
  dugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("last_newton_solution", duvalues_);
  edc.GetGradsState("last_newton_solution", dugrads_);

  edc.GetValuesState("last_time_solution", last_timestep_uvalues_);

  // changed
  qvalues_.reinit(1);
  edc.GetParamValues("control", qvalues_);

  std::vector<Tensor<1, 2>> phi_u(n_dofs_per_element);
  std::vector<Tensor<2, 2>> phi_grads_u(n_dofs_per_element);
  std::vector<double> div_phi_u(n_dofs_per_element);
  std::vector<double> phi_pf(n_dofs_per_element);
  std::vector<Tensor<1, 2>> phi_grads_pf(n_dofs_per_element);

  Tensor<2, 2> Identity;
  Identity[0][0] = 1.0;
  Identity[1][1] = 1.0;

  Tensor<2, 2> zero_matrix;
  zero_matrix.clear();

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    for (unsigned int k = 0; k < n_dofs_per_element; k++)
    {

      phi_u[k] = state_fe_values[displacements].value(k, q_point);
      phi_grads_u[k] = state_fe_values[displacements].gradient(k, q_point);
      div_phi_u[k] = state_fe_values[displacements].divergence(k, q_point);
      phi_pf[k] = state_fe_values[phasefield].value(k, q_point);
      phi_grads_pf[k] = state_fe_values[phasefield].gradient(k, q_point);
    }

    Tensor<2, 2> grad_u;
    grad_u.clear();
    grad_u[0][0] = ugrads_[q_point][0][0];
    grad_u[0][1] = ugrads_[q_point][0][1];
    grad_u[1][0] = ugrads_[q_point][1][0];
    grad_u[1][1] = ugrads_[q_point][1][1];

    Tensor<1, 2> v;
    v[0] = uvalues_[q_point](0);
    v[1] = uvalues_[q_point](1);

    Tensor<1, 2> grad_pf;
    grad_pf.clear();
    grad_pf[0] = ugrads_[q_point][2][0];
    grad_pf[1] = ugrads_[q_point][2][1];

    double pf = uvalues_[q_point](2);
    double old_timestep_pf = last_timestep_uvalues_[q_point](2);

    const Tensor<2, 2> E = 0.5 * (grad_u + transpose(grad_u));
    const double tr_E = trace(E);

    Tensor<2, 2> stress_term;
    stress_term.clear();
    stress_term = lame_coefficient_lambda_ * tr_E * Identity + 2 * lame_coefficient_mu_ * E;

    Tensor<2, 2> stress_term_plus;
    Tensor<2, 2> stress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(stress_term_plus, stress_term_minus,
                       E, tr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, false);
    }
    else
    {
      stress_term_plus = stress_term;
      stress_term_minus = 0;
    }

    // Prepare du for use

    // First the displacement part of du
    Tensor<2, 2> grad_duDisp;
    grad_duDisp.clear();
    grad_duDisp[0][0] = dugrads_[q_point][0][0];
    grad_duDisp[0][1] = dugrads_[q_point][0][1];
    grad_duDisp[1][0] = dugrads_[q_point][1][0];
    grad_duDisp[1][1] = dugrads_[q_point][1][1];
    // Next phase field part
    Tensor<1, 2> grad_duPf;
    grad_duPf.clear();
    grad_duPf[0] = dugrads_[q_point][2][0];
    grad_duPf[1] = dugrads_[q_point][2][1];
    // Lastly gradients of du coresponding to the multiplier
    Tensor<1, 2> grad_duMult;
    grad_duMult.clear();
    grad_duMult[0] = dugrads_[q_point][3][0];
    grad_duMult[1] = dugrads_[q_point][3][1];

    // prepare duvalues as well
    Tensor<2, 1> duDisp;
    duDisp[0] = duvalues_[q_point][0];
    duDisp[1] = duvalues_[q_point][1];
    double duPf = duvalues_[q_point][2];
    double duMult = duvalues_[q_point][3];

    // Now calculate E_lin and sigma + and - for du
    const Tensor<2, 2> duE = 0.5 * (grad_duDisp + transpose(grad_duDisp));
    const double dutr_E = trace(duE);

    Tensor<2, 2> dustress_term;
    dustress_term.clear();
    dustress_term = lame_coefficient_lambda_ * dutr_E * Identity + 2 * lame_coefficient_mu_ * duE;

    Tensor<2, 2> dustress_term_plus;
    Tensor<2, 2> dustress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(dustress_term_plus, dustress_term_minus,
                       duE, dutr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, false); // false as u only appears in sigma not in deriv sigma
    }
    else
    {
      dustress_term_plus = stress_term;
      dustress_term_minus = 0;
    }

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
    {

      const Tensor<2, 2> E_LinU = 0.5 * (phi_grads_u[i] + transpose(phi_grads_u[i])); // phi_grads_u are gradients of displacement testfuctions, used here because E_lin is evaluated in these displacements instead of in u

      const double tr_E_LinU = trace(E_LinU);

      Tensor<2, 2> stress_term_LinU;
      stress_term_LinU = lame_coefficient_lambda_ * tr_E_LinU * Identity + 2 * lame_coefficient_mu_ * E_LinU;

      Tensor<2, 2> stress_term_plus_LinU;
      Tensor<2, 2> stress_term_minus_LinU;

      // Necessary because stress splitting does not work
      // in the very initial time step.
      if (this->GetTime() > 0.001)
      {
        decompose_stress(stress_term_plus_LinU, stress_term_minus_LinU,
                         E, tr_E, E_LinU, tr_E_LinU,
                         lame_coefficient_lambda_,
                         lame_coefficient_mu_,
                         true); // true, since FE members only appear in derivatives of sigma
      }
      else
      {
        stress_term_plus_LinU = stress_term_LinU;
        stress_term_minus_LinU = 0;
      }

      // Solid (time-lagged version)
      // This then derivative of first line of (16) to u
      local_vector(i) += scale * (scalar_product(((1 - constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) * // This is g
                                                     stress_term_plus_LinU,
                                                 duE)                           // du - stress_term_plus_LinU is the derivative sigma+ applied to phi
                                  + scalar_product(stress_term_minus_LinU, duE) // du - See above for what stress term minus is
                                  ) *
                         state_fe_values.JxW(q_point);

      // Phase-field
      local_vector(i) += scale * (
                                     // Main terms
                                     (1 - constant_k_) * (scalar_product(stress_term_plus_LinU, E) // E is E_lin(u)
                                                          + scalar_product(stress_term_plus, E_LinU)) *
                                         pf * duPf                                                                // du - stress term plus is sigma+(u), derivative second line of (16)
                                     + (1 - constant_k_) * scalar_product(stress_term_plus, E) * phi_pf[i] * duPf // d phi - first term second row of (16)
                                     + qvalues_[0] / (alpha_eps_)*phi_pf[i] * duPf                                // d phi - deriv. of second term second row
                                     + qvalues_[0] * alpha_eps_ * phi_grads_pf[i] * grad_duPf                     // d phi - deriv of third term second row
                                     ) *
                         state_fe_values.JxW(q_point);

      // Now the Multiplierpart
      // only in vertices, so we check whether one of the
      // lambda test function
      //  is one (i.e. we are in a vertex)
      if (
          (fabs(state_fe_values[multiplier].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon()))
      {
        // Weight to account for multiplicity when running over multiple meshes.
        unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
        double weight = 1. / n_neig;

        if (n_neig == 4)
        {
          // max = 0
          if ((uvalues_[q_point][3] + s_ * (pf - old_timestep_pf)) <= 0.)
          {
            local_vector(i) += scale * weight * state_fe_values[multiplier].value(i, q_point) * duMult;
          }
          else // max > 0
          {
            // From Complementarity
            local_vector(i) -= scale * weight * s_ * state_fe_values[phasefield].value(i, q_point) * duMult;
          }

          for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            if (fabs(state_fe_values[multiplier].value(j, q_point) - 1.) < std::numeric_limits<double>::epsilon())
            {
              // From Equation
              local_vector(j) += scale * weight * duPf * state_fe_values[multiplier].value(i, q_point); // TODO just think about this again lol
            }
          }
        }
        else // Boundary or hanging node no weight so it works when hanging
        {
          local_vector(i) += scale * state_fe_values[multiplier].value(i, q_point) * duMult; // TODO Check this
        }
      }
    }
  }
}
// Domain values for elements
void ElementEquation_UTT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
{
  assert(this->problem_type_ == "adjoint_hessian");
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
  unsigned int n_q_points = edc.GetNQPoints();

  const FEValuesExtractors::Vector displacements(0);
  const FEValuesExtractors::Scalar phasefield(2);
  const FEValuesExtractors::Scalar multiplier(3);

  uvalues_.resize(n_q_points, Vector<double>(4));
  ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));

  edc.GetValuesState("state", uvalues_);
  edc.GetGradsState("state", ugrads_);

  dzvalues_.resize(n_q_points, Vector<double>(4));
  dzgrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("adjoint_hessian", dzvalues_);
  edc.GetGradsState("adjoint_hessian", dzgrads_);

  edc.GetValuesState("last_time_solution", last_timestep_uvalues_);

  // changed
  qvalues_.reinit(1);
  edc.GetParamValues("control", qvalues_);

  std::vector<Tensor<1, 2>> phi_u(n_dofs_per_element);
  std::vector<Tensor<2, 2>> phi_grads_u(n_dofs_per_element);
  std::vector<double> div_phi_u(n_dofs_per_element);
  std::vector<double> phi_pf(n_dofs_per_element);
  std::vector<Tensor<1, 2>> phi_grads_pf(n_dofs_per_element);

  Tensor<2, 2> Identity;
  Identity[0][0] = 1.0;
  Identity[1][1] = 1.0;

  Tensor<2, 2> zero_matrix;
  zero_matrix.clear();

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    for (unsigned int k = 0; k < n_dofs_per_element; k++)
    {

      phi_u[k] = state_fe_values[displacements].value(k, q_point);
      phi_grads_u[k] = state_fe_values[displacements].gradient(k, q_point);
      div_phi_u[k] = state_fe_values[displacements].divergence(k, q_point);
      phi_pf[k] = state_fe_values[phasefield].value(k, q_point);
      phi_grads_pf[k] = state_fe_values[phasefield].gradient(k, q_point);
    }

    Tensor<2, 2> grad_u;
    grad_u.clear();
    grad_u[0][0] = ugrads_[q_point][0][0];
    grad_u[0][1] = ugrads_[q_point][0][1];
    grad_u[1][0] = ugrads_[q_point][1][0];
    grad_u[1][1] = ugrads_[q_point][1][1];

    Tensor<1, 2> v;
    v[0] = uvalues_[q_point](0);
    v[1] = uvalues_[q_point](1);

    Tensor<1, 2> grad_pf;
    grad_pf.clear();
    grad_pf[0] = ugrads_[q_point][2][0];
    grad_pf[1] = ugrads_[q_point][2][1];

    double pf = uvalues_[q_point](2);
    double old_timestep_pf = last_timestep_uvalues_[q_point](2);

    const Tensor<2, 2> E = 0.5 * (grad_u + transpose(grad_u));
    const double tr_E = trace(E);

    Tensor<2, 2> stress_term;
    stress_term.clear();
    stress_term = lame_coefficient_lambda_ * tr_E * Identity + 2 * lame_coefficient_mu_ * E;

    Tensor<2, 2> stress_term_plus;
    Tensor<2, 2> stress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(stress_term_plus, stress_term_minus,
                       E, tr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, false); // false as u terms only appear in sigma, not in deriv sigma
    }
    else
    {
      stress_term_plus = stress_term;
      stress_term_minus = 0;
    }

    // Prepare dzgrads for use by splitting it into its components

    // First the displacement part of dz
    Tensor<2, 2> grad_dzDisp;
    grad_dzDisp.clear();
    grad_dzDisp[0][0] = dzgrads_[q_point][0][0];
    grad_dzDisp[0][1] = dzgrads_[q_point][0][1];
    grad_dzDisp[1][0] = dzgrads_[q_point][1][0];
    grad_dzDisp[1][1] = dzgrads_[q_point][1][1];
    // Next phase field part
    Tensor<1, 2> grad_dzPf;
    grad_dzPf.clear();
    grad_dzPf[0] = dzgrads_[q_point][2][0];
    grad_dzPf[1] = dzgrads_[q_point][2][1];
    // Lastly gradients of dz coresponding to the multiplier
    Tensor<1, 2> grad_dzMult;
    grad_dzMult.clear();
    grad_dzMult[0] = dzgrads_[q_point][3][0];
    grad_dzMult[1] = dzgrads_[q_point][3][1];

    // prepare dzvalues as well
    Tensor<2, 1> dzDisp;
    dzDisp[0] = dzvalues_[q_point][0];
    dzDisp[1] = dzvalues_[q_point][1];
    double dzPf = dzvalues_[q_point][2];
    double dzMult = dzvalues_[q_point][3];

    // Now calculate E_lin and sigma + and - for dz
    const Tensor<2, 2> dzE = 0.5 * (grad_dzDisp + transpose(grad_dzDisp));
    const double dztr_E = trace(dzE);

    Tensor<2, 2> dzstress_term;
    dzstress_term.clear();
    dzstress_term = lame_coefficient_lambda_ * dztr_E * Identity + 2 * lame_coefficient_mu_ * dzE;

    Tensor<2, 2> dzstress_term_plus;
    Tensor<2, 2> dzstress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(dzstress_term_plus, dzstress_term_minus,
                       dzE, dztr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, true); // true as dz terms only appear in deriv sigma not in sigma itself
    }
    else
    {
      dzstress_term_plus = stress_term;
      dzstress_term_minus = 0;
    }

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
    {
      // Solid (time-lagged version)
      local_vector(i) += scale * (scalar_product(((1 - constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) * dzstress_term_plus, phi_grads_u[i]) // du
                                  + scalar_product(dzstress_term_minus, phi_grads_u[i])                                                                      // du
                                  ) *
                         state_fe_values.JxW(q_point);

      // Phase-field
      local_vector(i) += scale * (
                                     // Main terms
                                     (1 - constant_k_) * (scalar_product(dzstress_term_plus, E) + scalar_product(stress_term_plus, dzE)) * pf * phi_pf[i] // du
                                     + (1 - constant_k_) * scalar_product(stress_term_plus, E) * phi_pf[i] * dzPf                                         // d phi
                                     + qvalues_[0] / (alpha_eps_)*phi_pf[i] * dzPf                                                                        // d phi
                                     + qvalues_[0] * alpha_eps_ * phi_grads_pf[i] * grad_dzPf                                                             // d phi
                                     ) *
                         state_fe_values.JxW(q_point);

      // Now the Multiplierpart
      // only in vertices, so we check whether one of the
      // lambda test function
      //  is one (i.e. we are in a vertex)
      if (
          (fabs(state_fe_values[multiplier].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon()))
      {
        // Weight to account for multiplicity when running over multiple meshes.
        unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
        double weight = 1. / n_neig;

        if (n_neig == 4)
        {
          // max = 0
          if ((uvalues_[q_point][3] + s_ * (pf - old_timestep_pf)) <= 0.)
          {
            local_vector(i) += scale * weight * state_fe_values[multiplier].value(i, q_point) * dzMult;
          }
          else // max > 0
          {
            // From Complementarity
            local_vector(i) -= scale * weight * s_ * dzPf * state_fe_values[multiplier].value(i, q_point);
          }
          // From Equation
          local_vector(i) += scale * weight * state_fe_values[phasefield].value(i, q_point) * dzMult;
        }
        else // Boundary or hanging node no weight so it works when hanging
        {
          local_vector(i) += scale * state_fe_values[multiplier].value(i, q_point) * dzMult; // non-derivative is qvalue, but that is actually <Lambda, multFE_i>, but letter is 1 in this bracket and hence does not show up in elementequation. Derive to lambda and seting in multFE_j gives element matrix, so this should be <zmult, multFE_i>
        }
      }
    }
  }
}

// changed
//  Domain values for elements
void ElementEquation_UU(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
{
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
  unsigned int n_q_points = edc.GetNQPoints();

  const FEValuesExtractors::Vector displacements(0);
  const FEValuesExtractors::Scalar phasefield(2);
  const FEValuesExtractors::Scalar multiplier(3);

  assert(this->problem_type_ == "adjoint_hessian");

  // initialization of state and gradients thereof
  uvalues_.resize(n_q_points, Vector<double>(4));
  ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("state", uvalues_);
  edc.GetGradsState("state", ugrads_);

  // initilization of z and gradient thereof
  zvalues_.resize(n_q_points, Vector<double>(4));
  zgrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("adjoint", zvalues_);
  edc.GetGradsState("adjoint", zgrads_);

  // initilization of du and gradient thereof
  duvalues_.resize(n_q_points, Vector<double>(4));
  dugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("tangent", duvalues_);
  edc.GetGradsState("tangent", dugrads_);

  // declaration of state finite elements and gradients of them
  std::vector<Tensor<2, 2>> phi_grads_u(n_dofs_per_element);
  std::vector<double> phi_pf(n_dofs_per_element);

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {

    for (unsigned int k = 0; k < n_dofs_per_element; k++)
    {
      phi_grads_u[k] = state_fe_values[displacements].gradient(k, q_point);
      phi_pf[k] = state_fe_values[phasefield].value(k, q_point);
    }

    Tensor<2, 2> grad_u;
    grad_u.clear();
    grad_u[0][0] = ugrads_[q_point][0][0];
    grad_u[0][1] = ugrads_[q_point][0][1];
    grad_u[1][0] = ugrads_[q_point][1][0];
    grad_u[1][1] = ugrads_[q_point][1][1];

    double pf = uvalues_[q_point](2);
    
    Tensor<2, 2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;
    
    Tensor<2, 2> zero_matrix;
    zero_matrix.clear();
    
    const Tensor<2, 2> E = 0.5 * (grad_u + transpose(grad_u));
    const double tr_E = trace(E);

    Tensor<2, 2> stress_term;
    stress_term.clear();
    stress_term = lame_coefficient_lambda_ * tr_E * Identity + 2 * lame_coefficient_mu_ * E;

    Tensor<2, 2> stress_term_plus;
    Tensor<2, 2> stress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(stress_term_plus, stress_term_minus,
                       E, tr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, false); // false as only appears in sigma
    }
    else
    {
      stress_term_plus = stress_term;
      stress_term_minus = 0;
    }

    // Prepare z for use
    // The displacement part of z
    Tensor<2, 2> grad_zDisp;
    grad_zDisp.clear();
    grad_zDisp[0][0] = zgrads_[q_point][0][0];
    grad_zDisp[0][1] = zgrads_[q_point][0][1];
    grad_zDisp[1][0] = zgrads_[q_point][1][0];
    grad_zDisp[1][1] = zgrads_[q_point][1][1];

    // prepare zvalues as well
    double zPf = zvalues_[q_point][2];

    // Now calculate E_lin and sigma + and - for z
    const Tensor<2, 2> zE = 0.5 * (grad_zDisp + transpose(grad_zDisp));
    const double ztr_E = trace(zE);

    Tensor<2, 2> zstress_term;
    zstress_term.clear();
    zstress_term = lame_coefficient_lambda_ * ztr_E * Identity + 2 * lame_coefficient_mu_ * zE;

    Tensor<2, 2> zstress_term_plus;
    Tensor<2, 2> zstress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(zstress_term_plus, zstress_term_minus,
                       zE, ztr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, true); // true as z components only appear in deriv sigma
    }
    else
    {
      zstress_term_plus = stress_term;
      zstress_term_minus = 0;
    }

    // Prepare du for use
    // Gradients of du coresponding to the multiplier
    Tensor<1, 2> grad_duMult;
    grad_duMult.clear();
    grad_duMult[0] = dugrads_[q_point][3][0];
    grad_duMult[1] = dugrads_[q_point][3][1];

    // prepare duvalues as well
    double duPf = duvalues_[q_point][2];

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
    {
      // Prepare the sigma and E for the FE (For the displacement part)
      const Tensor<2, 2> E_LinU = 0.5 * (phi_grads_u[i] + transpose(phi_grads_u[i]));

      const double tr_E_LinU = trace(E_LinU);

      Tensor<2, 2> stress_term_LinU;
      stress_term_LinU = lame_coefficient_lambda_ * tr_E_LinU * Identity + 2 * lame_coefficient_mu_ * E_LinU;

      Tensor<2, 2> stress_term_plus_LinU;
      Tensor<2, 2> stress_term_minus_LinU;

      // Necessary because stress splitting does not work
      // in the very initial time step.
      if (this->GetTime() > 0.001)
      {
        decompose_stress(stress_term_plus_LinU, stress_term_minus_LinU,
                         E, tr_E, E_LinU, tr_E_LinU,
                         lame_coefficient_lambda_,
                         lame_coefficient_mu_,
                         true); // Ok as FE elements only appear in deriv of sigma in this
      }
      else
      {
        stress_term_plus_LinU = stress_term_LinU;
        stress_term_minus_LinU = 0;
      }

      local_vector(i) += scale * (1 - constant_k_) * ((pf * (scalar_product(stress_term_plus_LinU, zE) + scalar_product(stress_term_plus_LinU, E_LinU)) * duPf) + (phi_pf[i] * (scalar_product(zstress_term_plus, E) + scalar_product(stress_term_plus, zE)) * duPf) + (zPf * (scalar_product(stress_term_plus_LinU, E) + scalar_product(stress_term_plus, E_LinU)) * duPf)) * state_fe_values.JxW(q_point);
    }
  }
}

void ElementEquation_Q(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
{
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_q_points = edc.GetNQPoints();

  uvalues_.resize(n_q_points, Vector<double>(4));
  ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("state", uvalues_);
  edc.GetGradsState("state", ugrads_);

  assert(this->problem_type_ == "gradient");
  zvalues_.resize(n_q_points, Vector<double>(4));
  zgrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("adjoint", zvalues_);
  edc.GetGradsState("adjoint", zgrads_);

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    Tensor<1, 2> grad_pf;
    grad_pf.clear();
    grad_pf[0] = ugrads_[q_point][2][0];
    grad_pf[1] = ugrads_[q_point][2][1];

    double pf = uvalues_[q_point](2);

    // Prepare zgrads for use
    // Phase field part
    Tensor<1, 2> grad_zPf;
    grad_zPf.clear();
    grad_zPf[0] = zgrads_[q_point][2][0];
    grad_zPf[1] = zgrads_[q_point][2][1];

    // prepare zvalues as well
    double zPf = zvalues_[q_point][2];

    local_vector(0) += scale * ((pf - 1) * zPf / alpha_eps_ // This is -1/alepha_eps * (1-pf) * z_pf just rearranged
                                +alpha_eps_ * grad_pf * grad_zPf) *
                       state_fe_values.JxW(q_point);
  }
}
void ElementEquation_QT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
{
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
  unsigned int n_q_points = edc.GetNQPoints();

  uvalues_.resize(n_q_points, Vector<double>(4));
  ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("state", uvalues_);
  edc.GetGradsState("state", ugrads_);

  const FEValuesExtractors::Vector displacements(0);
  const FEValuesExtractors::Scalar phasefield(2);
  const FEValuesExtractors::Scalar multiplier(3);

  dqvalues_.reinit(1);
  edc.GetParamValues("dq", dqvalues_);

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    // Prepare the phase-field values
    Tensor<1, 2> grad_pf;
    grad_pf.clear();
    grad_pf[0] = ugrads_[q_point][2][0];
    grad_pf[1] = ugrads_[q_point][2][1];
    double pf = uvalues_[q_point](2);

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
    {
      // Prepare the phase field finite elements
      const double phi_i_pf = state_fe_values[phasefield].value(i, q_point);
      const Tensor<1, 2> phi_i_grads_pf = state_fe_values[phasefield].gradient(i, q_point);

      local_vector(i) +=
          scale * (-1 / alpha_eps_ * (1 - pf) * phi_i_pf + alpha_eps_ * grad_pf * phi_i_grads_pf) * dqvalues_[0] * state_fe_values.JxW(q_point);
    }
  }
}
void ElementEquation_QTT(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
{
  assert(this->problem_type_ == "hessian");
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_q_points = edc.GetNQPoints();

  uvalues_.resize(n_q_points, Vector<double>(4));
  ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("state", uvalues_);
  edc.GetGradsState("state", ugrads_);

  dzvalues_.resize(n_q_points, Vector<double>(4));
  dzgrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("adjoint_hessian", dzvalues_);
  edc.GetGradsState("adjoint_hessian", dzgrads_);

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    Tensor<1, 2> grad_pf;
    grad_pf.clear();
    grad_pf[0] = ugrads_[q_point][2][0];
    grad_pf[1] = ugrads_[q_point][2][1];

    double pf = uvalues_[q_point](2);

    // Prepare dzgrads for use by splitting it into its components

    // Phase field part of dz
    Tensor<1, 2> grad_dzPf;
    grad_dzPf.clear();
    grad_dzPf[0] = dzgrads_[q_point][2][0];
    grad_dzPf[1] = dzgrads_[q_point][2][1];
    // Gradients of dz coresponding to the multiplier
    Tensor<1, 2> grad_dzMult;
    grad_dzMult.clear();
    grad_dzMult[0] = dzgrads_[q_point][3][0];
    grad_dzMult[1] = dzgrads_[q_point][3][1];

    // prepare dzvalues as well
    double dzPf = dzvalues_[q_point][2];

    local_vector(0) += scale * ((pf - 1) * dzPf / alpha_eps_ // This is -1/alepha_eps * (1-pf) * z_pf just rearranged
                                +alpha_eps_ * grad_pf * grad_dzPf) *
                       state_fe_values.JxW(q_point);

  }
}
void ElementEquation_QU(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
{
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
  unsigned int n_q_points = edc.GetNQPoints();

  // initilization of z and gradient thereof
  zvalues_.resize(n_q_points, Vector<double>(4));
  zgrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("adjoint", zvalues_); // TODO should this be adjoint?
  edc.GetGradsState("adjoint", zgrads_);

  // initilization of dq
  dqvalues_.reinit(1);
  edc.GetParamValues("dq", dqvalues_); // TODO could be wrong and getValues... instead

  // Extractors for the FE
  const FEValuesExtractors::Vector displacements(0);
  const FEValuesExtractors::Scalar phasefield(2);
  const FEValuesExtractors::Scalar multiplier(3);

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    // Prepare z for use
    // The phase field part
    Tensor<1, 2> grad_zPf;
    grad_zPf.clear();
    grad_zPf[0] = zgrads_[q_point][2][0];
    grad_zPf[1] = zgrads_[q_point][2][1];

    // prepare zvalues as well
    double zPf = zvalues_[q_point][2];

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
    {
      const double phi_i_pf = state_fe_values[phasefield].value(i, q_point);
      const Tensor<1, 2> phi_i_grads_pf = state_fe_values[phasefield].gradient(i, q_point);

      local_vector(i) += scale * ((1 / alpha_eps_) * phi_i_pf * zPf + alpha_eps_ * phi_i_grads_pf * grad_zPf) * dqvalues_[0] * state_fe_values.JxW(q_point);
    }
  }
}

void ElementEquation_UQ(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
{
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_q_points = edc.GetNQPoints();

  // initilization of z and gradient thereof
  zvalues_.resize(n_q_points, Vector<double>(4));
  zgrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("adjoint", zvalues_);
  edc.GetGradsState("adjoint", zgrads_);

  // initilization of du and gradient thereof
  duvalues_.resize(n_q_points, Vector<double>(4));
  dugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  edc.GetValuesState("tangent", duvalues_);
  edc.GetGradsState("tangent", dugrads_);

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    // Prepares gradies of z
    // The phase field part
    Tensor<1, 2> grad_zPf;
    grad_zPf.clear();
    grad_zPf[0] = zgrads_[q_point][2][0];
    grad_zPf[1] = zgrads_[q_point][2][1];

    // prepare zvalues as well
    double zPf = zvalues_[q_point][2];

    // Prepare du for use
    // The phase field part
    Tensor<1, 2> grad_duPf;
    grad_duPf.clear();
    grad_duPf[0] = dugrads_[q_point][2][0];
    grad_duPf[1] = dugrads_[q_point][2][1];

    // prepare duvalues as well
    double duPf = duvalues_[q_point][2];

    local_vector(0) += scale * ((1 / alpha_eps_) * duPf * zPf + alpha_eps_ * grad_duPf * grad_zPf) * state_fe_values.JxW(q_point);
  }
}
void ElementEquation_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                        dealii::Vector<double> & /*local_vector*/, double /*scale*/,
                        double /*scale_ico*/)
{
  assert(this->problem_type_ == "hessian");
}

void ElementMatrix(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_matrix, double scale, double)
{
  const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
  unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
  unsigned int n_q_points = edc.GetNQPoints();

  const FEValuesExtractors::Vector displacements(0);
  const FEValuesExtractors::Scalar phasefield(2);
  const FEValuesExtractors::Scalar multiplier(3);

  uvalues_.resize(n_q_points, Vector<double>(4));
  ugrads_.resize(n_q_points, vector<Tensor<1, 2>>(4));
  last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));

  edc.GetValuesState("last_newton_solution", uvalues_);
  edc.GetGradsState("last_newton_solution", ugrads_);

  edc.GetValuesState("last_time_solution", last_timestep_uvalues_);

  // changed
  qvalues_.reinit(1);
  edc.GetParamValues("control", qvalues_);

  std::vector<Tensor<1, 2>> phi_u(n_dofs_per_element);
  std::vector<Tensor<2, 2>> phi_grads_u(n_dofs_per_element);
  std::vector<double> div_phi_u(n_dofs_per_element);
  std::vector<double> phi_pf(n_dofs_per_element);
  std::vector<Tensor<1, 2>> phi_grads_pf(n_dofs_per_element);

  Tensor<2, 2> Identity;
  Identity[0][0] = 1.0;
  Identity[1][1] = 1.0;

  Tensor<2, 2> zero_matrix;
  zero_matrix.clear();

  for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
  {
    for (unsigned int k = 0; k < n_dofs_per_element; k++)
    {

      phi_u[k] = state_fe_values[displacements].value(k, q_point);
      phi_grads_u[k] = state_fe_values[displacements].gradient(k, q_point);
      div_phi_u[k] = state_fe_values[displacements].divergence(k, q_point);
      phi_pf[k] = state_fe_values[phasefield].value(k, q_point);
      phi_grads_pf[k] = state_fe_values[phasefield].gradient(k, q_point);
    }

    Tensor<2, 2> grad_u;
    grad_u.clear();
    grad_u[0][0] = ugrads_[q_point][0][0];
    grad_u[0][1] = ugrads_[q_point][0][1];
    grad_u[1][0] = ugrads_[q_point][1][0];
    grad_u[1][1] = ugrads_[q_point][1][1];

    Tensor<1, 2> v;
    v[0] = uvalues_[q_point](0);
    v[1] = uvalues_[q_point](1);

    Tensor<1, 2> grad_pf;
    grad_pf.clear();
    grad_pf[0] = ugrads_[q_point][2][0];
    grad_pf[1] = ugrads_[q_point][2][1];

    double pf = uvalues_[q_point](2);
    double old_timestep_pf = last_timestep_uvalues_[q_point](2);

    const Tensor<2, 2> E = 0.5 * (grad_u + transpose(grad_u));
    const double tr_E = trace(E);

    Tensor<2, 2> stress_term;
    stress_term.clear();
    stress_term = lame_coefficient_lambda_ * tr_E * Identity + 2 * lame_coefficient_mu_ * E;

    Tensor<2, 2> stress_term_plus;
    Tensor<2, 2> stress_term_minus;

    // Necessary because stress splitting does not work
    // in the very initial time step.
    if (this->GetTime() > 0.001)
    {
      decompose_stress(stress_term_plus, stress_term_minus,
                       E, tr_E, zero_matrix, 0.0,
                       lame_coefficient_lambda_,
                       lame_coefficient_mu_, false);
    }
    else
    {
      stress_term_plus = stress_term;
      stress_term_minus = 0;
    }

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
    {

      const Tensor<2, 2> E_LinU = 0.5 * (phi_grads_u[i] + transpose(phi_grads_u[i])); // phi_grads_u are gradients of displacement testfuctions, used here because E_lin is evaluated in these displacements instead of in u

      const double tr_E_LinU = trace(E_LinU);

      Tensor<2, 2> stress_term_LinU;
      stress_term_LinU = lame_coefficient_lambda_ * tr_E_LinU * Identity + 2 * lame_coefficient_mu_ * E_LinU;

      Tensor<2, 2> stress_term_plus_LinU;
      Tensor<2, 2> stress_term_minus_LinU;

      // Necessary because stress splitting does not work
      // in the very initial time step.
      if (this->GetTime() > 0.001)
      {
        decompose_stress(stress_term_plus_LinU, stress_term_minus_LinU,
                         E, tr_E, E_LinU, tr_E_LinU,
                         lame_coefficient_lambda_,
                         lame_coefficient_mu_,
                         true);
      }
      else
      {
        stress_term_plus_LinU = stress_term_LinU;
        stress_term_minus_LinU = 0;
      }

      for (unsigned int j = 0; j < n_dofs_per_element; j++)
      {
        // Solid (time-lagged version)
        // This then derivative of first line of (16) to u
        local_matrix(j, i) += scale * (scalar_product(((1 - constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) * // This is g
                                                          stress_term_plus_LinU,
                                                      phi_grads_u[j])                           // du - stress_term_plus_LinU is the derivative sigma+ applied to phi
                                       + scalar_product(stress_term_minus_LinU, phi_grads_u[j]) // du - See above for what stress term minus is
                                       ) *
                              state_fe_values.JxW(q_point);

        // Phase-field
        local_matrix(j, i) += scale * (
                                          // Main terms
                                          (1 - constant_k_) * (scalar_product(stress_term_plus_LinU, E) // E is E_lin(u)
                                                               + scalar_product(stress_term_plus, E_LinU)) *
                                              pf * phi_pf[j]                                                                // du - stress term plus is sigma+(u), derivative second line of (16)
                                          + (1 - constant_k_) * scalar_product(stress_term_plus, E) * phi_pf[i] * phi_pf[j] // d phi - first term second row of (16)
                                          + qvalues_[0] / (alpha_eps_)*phi_pf[i] * phi_pf[j]                                // d phi - deriv. of second term second row
                                          + qvalues_[0] * alpha_eps_ * phi_grads_pf[i] * phi_grads_pf[j]                    // d phi - deriv of third term second row
                                          ) *
                              state_fe_values.JxW(q_point);

        // Now the Multiplierpart
        // only in vertices, so we check whether one of the
        // lambda test function
        //  is one (i.e. we are in a vertex)
        if (
            (fabs(state_fe_values[multiplier].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon()) ||
            (fabs(state_fe_values[multiplier].value(j, q_point) - 1.) < std::numeric_limits<double>::epsilon()))
        {
          // Weight to account for multiplicity when running over multiple meshes.
          unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
          double weight = 1. / n_neig;

          if (n_neig == 4)
          {
            // max = 0
            if ((uvalues_[q_point][3] + s_ * (pf - old_timestep_pf)) <= 0.) // This the max in (16)
            {
              local_matrix(i, j) += scale * weight * state_fe_values[multiplier].value(i, q_point) * state_fe_values[multiplier].value(j, q_point); // This is (1-G)(phi^j, phi^i)
            }
            else // max > 0
            {
              // From Complementarity
              local_matrix(i, j) -= scale * weight * s_ * state_fe_values[phasefield].value(j, q_point) * state_fe_values[multiplier].value(i, q_point); // This is G*(psi^j, phi^i)
            }
            // From Equation
            local_matrix(i, j) += scale * weight * state_fe_values[phasefield].value(i, q_point) * state_fe_values[multiplier].value(j, q_point); // This just (phi^j, psi^i)
          }
          else // Boundary or hanging node no weight so it works when hanging
          {
            local_matrix(i, j) += scale * state_fe_values[multiplier].value(i, q_point) * state_fe_values[multiplier].value(j, q_point);
          }
        }
      }
    }
  }
}


// Edited
void ControlElementEquation(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale)
{
  {
    assert(
        (this->problem_type_ == "gradient") || (this->problem_type_ == "hessian"));
    funcgradvalues_.reinit(local_vector.size());
    edc.GetParamValues("last_newton_solution", funcgradvalues_);
  }

  for (unsigned int i = 0; i < local_vector.size(); i++)
  {
    local_vector(i) += scale * funcgradvalues_[i];
  }
}

void ControlElementMatrix(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_matrix, double scale)
{
  local_matrix(0, 0) = scale;
}

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> & /*local_vector*/,
                       double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }
  
  void
  ElementTimeEquationExplicit(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                              dealii::Vector<double> & /*local_vector*/,
                              double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeEquation(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                      dealii::Vector<double> & /*local_vector*/,
                      double /*scale*/)
  {
    assert(this->problem_type_ == "state");

  }

  void
  ElementTimeMatrixExplicit(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                            FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeMatrix(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");

  }


UpdateFlags
GetUpdateFlags() const
{
  if (this->problem_type_ == "state" || this->problem_type_ == "adjoint" || this->problem_type_ == "adjoint_hessian" || this->problem_type_ == "tangent")
    return update_values | update_gradients | update_quadrature_points;
  else if (this->problem_type_ == "gradient" || this->problem_type_ == "hessian")
    return update_values | update_quadrature_points;
  else
    throw DOpEException("Unknown Problem Type " + this->problem_type_,
                        "LocalPDE::GetUpdateFlags");
}

UpdateFlags
GetFaceUpdateFlags() const
{
  if (this->problem_type_ == "state" || this->problem_type_ == "adjoint" || this->problem_type_ == "adjoint_hessian" || this->problem_type_ == "tangent" || this->problem_type_ == "gradient" || this->problem_type_ == "hessian")
    return update_default;
  else
    throw DOpEException("Unknown Problem Type " + this->problem_type_,
                        "LocalPDE::GetFaceUpdateFlags");
}

unsigned int
GetControlNBlocks() const
{
  return 1;
}

unsigned int
GetStateNBlocks() const
{
  return 1;
}

std::vector<unsigned int> &
GetControlBlockComponent()
{
  return control_block_component_;
}
const std::vector<unsigned int> &
GetControlBlockComponent() const
{
  return control_block_component_;
}
std::vector<unsigned int> &
GetStateBlockComponent()
{
  return state_block_component_;
}
const std::vector<unsigned int> &
GetStateBlockComponent() const
{
  return state_block_component_;
}

bool
HasVertices() const
{
  return true;
}

private:
Vector<double> qvalues_; // This the control values, so G_c
Vector<double> dqvalues_;
Vector<double> funcgradvalues_;

// Changed
vector<Vector<double>> uvalues_;
vector<Vector<double>> zvalues_;
vector<Vector<double>> dzvalues_;
vector<Vector<double>> duvalues_;
vector<vector<Tensor<1, dealdim>>> ugrads_;
vector<vector<Tensor<1, dealdim>>> zgrads_;
vector<vector<Tensor<1, dealdim>>> dugrads_;
vector<vector<Tensor<1, dealdim>>> dzgrads_;



vector<Vector<double>> last_timestep_uvalues_;

vector<unsigned int> state_block_component_;
vector<unsigned int> control_block_component_;

double constant_k_, alpha_eps_, lame_coefficient_mu_, lame_coefficient_lambda_, s_;
}
;
#endif
