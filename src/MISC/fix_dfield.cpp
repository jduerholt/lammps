/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Christina Payne (Vanderbilt U)
                        Stan Moore (Sandia) for dipole terms
------------------------------------------------------------------------- */

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include "fix_dfield.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "modify.h"
#include "force.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "region.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL,ATOM};

#define INVOKED_SCALAR 1

/* ---------------------------------------------------------------------- */

FixDfield::FixDfield(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), xstr_d(NULL), ystr_d(NULL), zstr_d(NULL),
  xstr_p(NULL), ystr_p(NULL), zstr_p(NULL),
  estr(NULL), idregion(NULL), dfield(NULL)
{
  if (strcmp(update->unit_style,"real") != 0) error->warning(FLERR,"Energy computed with fix dfield only compatible with 'real' energy units");

  if (narg < 9) error->all(FLERR,"Illegal fix efield command");

  dynamic_group_allow = 1;
  vector_flag = 1;
  scalar_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extvector = 1;
  extscalar = 1;
  respa_level_support = 1;
  ilevel_respa = 0;
  //virial_flag = 1;

  dxflag = dyflag = dzflag = 1;

  qe2f = force->qe2f;
  xstr_d = ystr_d = zstr_d = NULL;

  if (strstr(arg[3],"v_") == arg[3]) {
    error->all(FLERR,"fix_dfield only constant displacement fields at the moment");
    int n = strlen(&arg[3][2]) + 1;
    xstr_d = new char[n];
    strcpy(xstr_d,&arg[3][2]);
  } else {
    if (strcmp(arg[3],"NULL") == 0){dxflag = 0; dx=0.0;} else {dx = force->numeric(FLERR,arg[3]);}
    xstyle = CONSTANT;
  }

  if (strstr(arg[4],"v_") == arg[4]) {
    error->all(FLERR,"fix_dfield only constant displacement fields at the moment");
    int n = strlen(&arg[4][2]) + 1;
    ystr_d = new char[n];
    strcpy(ystr_d,&arg[4][2]);
  } else {
    if (strcmp(arg[4],"NULL") == 0){dyflag = 0; dy=0.0;} else{dy = force->numeric(FLERR,arg[4]);}
    ystyle = CONSTANT;
  }

  if (strstr(arg[5],"v_") == arg[5]) {
    error->all(FLERR,"fix_dfield only constant displacement fields at the moment");
    int n = strlen(&arg[5][2]) + 1;
    zstr_d = new char[n];
    strcpy(zstr_d,&arg[5][2]);
  } else {
    if (strcmp(arg[5],"NULL") == 0){dzflag = 0; dz=0.0;} else{dz = force->numeric(FLERR,arg[5]);}
    zstyle = CONSTANT;
  }

  // SJC: the below three clauses get the name of the compute for the
  // polarization components.
  if (strstr(arg[6],"c_") == arg[6]) {
    int n = strlen(&arg[6][2]) + 1;
    xstr_p = new char[n];
    strcpy(xstr_p,&arg[6][2]);
  } else {
    error->all(FLERR,"Polarization in fix_dfield must be a compute.");
  }

  if (strstr(arg[7],"c_") == arg[7]) {
    int n = strlen(&arg[7][2]) + 1;
    ystr_p = new char[n];
    strcpy(ystr_p,&arg[7][2]);
  } else {
    error->all(FLERR,"Polarization in fix_dfield must be a compute.");
  }

  if (strstr(arg[8],"c_") == arg[8]) {
    int n = strlen(&arg[8][2]) + 1;
    zstr_p = new char[n];
    strcpy(zstr_p,&arg[8][2]);
  } else {
    error->all(FLERR,"Polarization in fix_dfield must be a compute.");
  }

  // optional args

  iregion = -1;
  idregion = NULL;
  estr = NULL;

  int iarg = 9;

//  while (iarg < narg) {
//    if (strcmp(arg[iarg],"region") == 0) {
//      error->all(FLERR,"fix_dfield not yet implemented with 'region' keyword");
//    } else if (strcmp(arg[iarg],"energy") == 0) {
//      error->all(FLERR,"fix_dfield not yet implemented with 'energy' keyword");
//    } else error->all(FLERR,"Illegal fix dfield command");
//  }
  /*while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix efield command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix efield does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix efield command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        int n = strlen(&arg[iarg+1][2]) + 1;
        estr = new char[n];
        strcpy(estr,&arg[iarg+1][2]);
      } else error->all(FLERR,"Illegal fix efield command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix efield command");
  }*/

  force_flag = 0;
  fsum[0] = fsum[1] = fsum[2] = fsum[3] = 0.0;

  maxatom = atom->nmax;
  memory->create(dfield,maxatom,4,"dfield:dfield");
}

/* ---------------------------------------------------------------------- */

FixDfield::~FixDfield()
{
  delete [] xstr_d;
  delete [] ystr_d;
  delete [] zstr_d;
  delete [] estr;
  delete [] idregion;
  memory->destroy(dfield);
}

/* ---------------------------------------------------------------------- */

int FixDfield::setmask()
{
  int mask = 0;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDfield::init()
{
  qflag = muflag = 0;
  if (atom->q_flag) qflag = 1;
  if (atom->mu_flag && atom->torque_flag) muflag = 1;
  if (!qflag && !muflag)
    error->all(FLERR,"Fix dfield requires atom attribute q or mu");

  if(muflag){error->warning(FLERR,"fix dfield not implemented for dipoles yet");}
  // check variables

  if (xstr_d) {
    xvar = input->variable->find(xstr_d);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix dfield does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
    else error->all(FLERR,"Variable for fix dfield is invalid style");
  }
  if (ystr_d) {
    yvar = input->variable->find(ystr_d);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix dfield does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
    else error->all(FLERR,"Variable for fix dfield is invalid style");
  }
  if (zstr_d) {
    zvar = input->variable->find(zstr_d);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix dfield does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
    else error->all(FLERR,"Variable for fix dfield is invalid style");
  }
  if (estr) {
    evar = input->variable->find(estr);
    if (evar < 0)
      error->all(FLERR,"Variable name for fix dfield does not exist");
    if (input->variable->atomstyle(evar)) estyle = ATOM;
    else error->all(FLERR,"Variable for fix dfield is invalid style");
  } else estyle = NONE;

  // SJC: 
  int iOmegaPx = modify->find_compute(xstr_p);
  int iOmegaPy = modify->find_compute(ystr_p);
  int iOmegaPz = modify->find_compute(zstr_p);

  if (iOmegaPx < 0 || iOmegaPy < 0 || iOmegaPz < 0)
    error->all(FLERR,"Could not find compute for polaraztion in fix_dfield");
  c_OmegaPx = modify->compute[iOmegaPx];
  c_OmegaPy = modify->compute[iOmegaPy];
  c_OmegaPz = modify->compute[iOmegaPz];



  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix aveforce does not exist");
  }

  if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

  if (muflag && varflag == ATOM)
    error->all(FLERR,"Fix dfield with dipoles cannot use atom-style variables");

  if (muflag && update->whichflag == 2 && comm->me == 0)
    error->warning(FLERR,
                   "The minimizer does not re-orient dipoles "
                   "when using fix dfield");

  if (varflag == CONSTANT && estyle != NONE)
    error->all(FLERR,"Cannot use variable energy with "
               "constant dfield in fix dfield");
  if ((varflag == EQUAL || varflag == ATOM) &&
      update->whichflag == 2 && estyle == NONE)
    error->all(FLERR,"Must use variable energy with fix dfield");

  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixDfield::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixDfield::min_setup(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   apply F = qE
------------------------------------------------------------------------- */

void FixDfield::post_force(int vflag)
{
  double **f = atom->f;
  double *q = atom->q;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  // energy and virial setup

  if (vflag) v_setup(vflag);
  else evflag = 0;

  // reallocate efield array if necessary

  if (varflag == ATOM && atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(dfield);
    memory->create(dfield,maxatom,4,"dfield:dfield");
  }

  // update region if necessary

  Region *region = NULL;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }

  // fsum[0] = "potential energy" for added force
  // fsum[123] = extra force added to atoms

  fsum[0] = fsum[1] = fsum[2] = fsum[3] = 0.0;
  force_flag = 0;
  

  // energy is the extra energy added by the const D hamiltonian
  //energy = 0;

  double **x = atom->x;
  double fx,fy,fz;
  double v[6];

  double dipole[3];
  double dipole_all[3];
  double pol[3];
  double D[3];
  double DmP[3];
  DmP[0] = DmP[1] = DmP[2] = 0.0;
  double volume = domain->xprd * domain->yprd * domain->zprd;
  double volumeinv = 1.0/volume;

  // constant dfield

  // we have to adjust D to the current lattice
  // ie compute D out of d and the cellparams

  if (!domain->triclinic){
    D[0]=dx*domain->xprd;
    D[1]=dy*domain->yprd;
    D[2]=dz*domain->zprd;
  } else {
    D[0]=dx*domain->h[0];
    D[1]=dx*domain->h[3]+dy*domain->h[1];
    D[2]=dx*domain->h[4]+dy*domain->h[5]+dz*domain->h[3];
  }
  D[0] = D[0]/volume;
  D[1] = D[1]/volume;
  D[2] = D[2]/volume;


  if (varflag == CONSTANT) {
    double unwrap[3];

    // SJC: do I need to worry about invoking? c.f. compute_heat_flux.cpp line 109
    modify->clearstep_compute(); // SJC: this is required to clear the
				 // invoked flags, otherwise the
				 // polarization isn't properly updated

    // SJC: Not sure if this is entirely needed but it seems to work.
    if (!(c_OmegaPx->invoked_flag & INVOKED_SCALAR)) {
      c_OmegaPx->compute_scalar();
      c_OmegaPx->invoked_flag |= INVOKED_SCALAR;
    }
    if (!(c_OmegaPy->invoked_flag & INVOKED_SCALAR)) {
      c_OmegaPy->compute_scalar();
      c_OmegaPy->invoked_flag |= INVOKED_SCALAR;
    }
    if (!(c_OmegaPz->invoked_flag & INVOKED_SCALAR)) {
      c_OmegaPz->compute_scalar();
      c_OmegaPz->invoked_flag |= INVOKED_SCALAR;
    }

    pol[0]=volumeinv*c_OmegaPx->scalar;
    pol[1]=volumeinv*c_OmegaPy->scalar;
    pol[2]=volumeinv*c_OmegaPz->scalar;

    //std::cout << pol[0] << std::endl;
    //std::cout << pol[1] << std::endl;
    //std::cout << pol[2] << std::endl;


    // calculate DmP, TODO loop over it
    if(dxflag){DmP[0] += D[0]-pol[0];}
    if(dyflag){DmP[1] += D[1]-pol[1];}
    if(dzflag){DmP[2] += D[2]-pol[2];}

    //std::cout << DmP[0] << std::endl;
    //std::cout << DmP[1] << std::endl;
    //std::cout << DmP[2] << std::endl;

    // now we can calculate the energy, works only for unit system real
    // eps0 in units of unitcharge/(V*Angstrom)
    // efact is faraday const *0.1/eps0
    double epsilon0 = 5.526348e-3;
    double efact = 0.239*96.4853082e0/epsilon0;
    ///double efact = 9648.53082e0/epsilon0;
    // energy is afterwards in kJ/mol

    fsum[0] = fsum[0] + volume/2.0e0*efact*(DmP[0]*DmP[0]);
    fsum[0] = fsum[0] + volume/2.0e0*efact*(DmP[1]*DmP[1]);
    fsum[0] = fsum[0] + volume/2.0e0*efact*(DmP[2]*DmP[2]);

    //std::cout <<fsum[0]<< std::endl;



    // now we have to compute the forces
    if (qflag){
      for (int i = 0; i < nlocal; i++){
        if (mask[i] & groupbit) {
          if(dxflag){fx = efact*q[i]*DmP[0];} else {fx=0.0;}
          if(dyflag){fy = efact*q[i]*DmP[1];} else {fy=0.0;}
          if(dzflag){fz = efact*q[i]*DmP[2];} else {fz=0.0;}

          f[i][0] += fx;
          f[i][1] += fy;
          f[i][2] += fz;
          //std::cout <<fx<< std::endl;
          //std::cout <<fy<< std::endl;
          //std::cout <<fz<< std::endl;
          domain->unmap(x[i],image[i],unwrap);

          fsum[1] += fx;
          fsum[2] += fy;
          fsum[3] += fz;
        }
      }
    }

    if (muflag){
      error->all(FLERR,"fix_dfield only charges at the moment");   
    }

  } else {
    error->all(FLERR,"fix_dfield only works with constant displacement fields at the moment");   
  }
}

/* ---------------------------------------------------------------------- */

void FixDfield::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixDfield::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixDfield::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = atom->nmax*4 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   return energy added by fix
------------------------------------------------------------------------- */

double FixDfield::compute_scalar(void)
{
  //std::cout << "mama" << std::endl;
  return fsum[0];
}

/* ----------------------------------------------------------------------
   return total extra force due to fix
------------------------------------------------------------------------- */

double FixDfield::compute_vector(int n)
{
  if (force_flag == 0) {
    MPI_Allreduce(fsum,fsum_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return fsum_all[n+1];
}

