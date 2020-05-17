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
   Contributing author: Richard Berger (Temple U)
------------------------------------------------------------------------- */

#include <Python.h>
//#include <numpy/arrayobject.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include "fix_python_invoke.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "lmppython.h"
#include "python_compat.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "domain.h"



using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixPythonInvoke::FixPythonInvoke(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  //if (narg != 6) error->all(FLERR,"Illegal fix python/invoke command");
  if (narg < 6) error->all(FLERR,"Illegal fix python/invoke command");
  if (narg > 8) error->all(FLERR,"Illegal fix python/invoke command");

  // Flags.
  scalar_flag = 1;
  
  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix python/invoke command");

  // ensure Python interpreter is initialized
  python->init();
  // array import and initialization not working, need to be fixed
  // import_array1();
  // _import_array();

  if (strcmp(arg[4],"post_force") == 0) {
    selected_callback = POST_FORCE;
  } else if (strcmp(arg[4],"end_of_step") == 0) {
    selected_callback = END_OF_STEP;
  } else {
    error->all(FLERR,"Unsupported callback name for fix python/invoke");
  }
  // RS call min_post_force in addition
  selected_callback |= MIN_POST_FORCE;
  selected_callback |= THERMO_ENERGY;

  // get Python function
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject * pyMain = PyImport_AddModule("__main__");

  if (!pyMain) {
    PyGILState_Release(gstate);
    error->all(FLERR,"Could not initialize embedded Python");
  }

  char * fname = arg[5];
  pFunc = PyObject_GetAttrString(pyMain, fname);

  if (!pFunc) {
    PyGILState_Release(gstate);
    error->all(FLERR,"Could not find Python function");
  }

  PyGILState_Release(gstate);

  // JPD setup neighborlist stuff
  // check if a fifth argument is provided
  if (narg>6){
    updateNlist = 1;
    if (narg != 8) error->all(FLERR,"Illegal fix python/invoke command");
    // setup second callback
/*    char * fname2 = arg[6];
    //std::cout << fname2 << std::endl;
    pNeighFunc = PyObject_GetAttrString(pyMain, fname2);
    if (!pNeighFunc) {
      PyGILState_Release(gstate);
      error->all(FLERR,"Could not find Python neighborlist function");
    } */
    nlist_max = force->inumeric(FLERR,arg[6]);
    // set cutoff
    maxCutoffRadius = force->numeric(FLERR,arg[7]);
    //std::cout << maxCutoffRadius << std::endl;
    nlist_mapping = (int *)calloc(sizeof(int),atom->natoms);
    nlist_neighbors = (int *)calloc(sizeof(int),nlist_max);
    nlist_offset = (int *)calloc(sizeof(int),nlist_max*3);
  } else {
    updateNlist = 0;
  }
}

void FixPythonInvoke::init()
{
  std::cout << "request nlist" << std::endl;
  if (updateNlist==1){
    int irequest = neighbor->request(this,instance_me);
    //int irequest = neighbor->request((void *) this);
    neighbor->requests[irequest]->pair=0;
    neighbor->requests[irequest]->fix=1;
    neighbor->requests[irequest]->half=0;
    neighbor->requests[irequest]->full=1;
    neighbor->requests[irequest]->cut=1;
    neighbor->requests[irequest]->occasional = 1;
    // add skin distance explicitely
    neighbor->requests[irequest]->cutoff=maxCutoffRadius+neighbor->skin;
    //neighbor->requests[irequest]->cutoff=maxCutoffRadius+2.0;
  }
}

/* ---------------------------------------------------------------------- */

int FixPythonInvoke::setmask()
{
  return selected_callback;
}

/* ---------------------------------------------------------------------- */

void FixPythonInvoke::end_of_step()
{
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject * ptr = PY_VOID_POINTER(lmp);
  PyObject * arglist = Py_BuildValue("(O)", ptr);

  PyObject * result = PyEval_CallObject((PyObject*)pFunc, arglist);
  Py_DECREF(arglist);

  PyGILState_Release(gstate);
}

// JPD: Not working
/*PyObject *makearray(int array[], size_t size) {
    std::cout << "make array called" << std::endl;
    npy_intp dims[1];// = size;
    std::cout << size << std::endl;
    std::cout << array[1] << std::endl;
    dims[0] = size;
    std::cout << "dims created" << std::endl;
    PyObject *a = PyArray_SimpleNewFromData(1,dims,NPY_INT, (void*)array);
    std::cout << "make array created" << std::endl;
    return a; 
}*/

// This is slow and leaks memory!!!
/*PyObject *makelist(int array[], size_t size) {
    PyObject *l = PyList_New(size);
    for (size_t i = 0; i != size; ++i) {
        PyList_SET_ITEM(l, i, PY_INT_FROM_LONG(array[i]));
        //PyList_SetItem(l, i, PY_INT_FROM_LONG(array[i]));
    }
    return l;
}*/

/* ---------------------------------------------------------------------- */

void FixPythonInvoke::post_force(int vflag)
{
  if (update->ntimestep % nevery != 0) return;

  PyGILState_STATE gstate = PyGILState_Ensure();

  // JPD perform nlist passing if necessary
  if (updateNlist==1){
    //std::cout<< "Pass Nlist"<< std::endl;
    //transferNeighborList();
    computeNeighborList();
  }

  // now evaluate the actual callback function
  PyObject * ptr = PY_VOID_POINTER(lmp);

  //PyObject * ptr_nlist_mapping = makelist(nlist_mapping, atom->natoms);
  //PyObject * ptr_nlist_neighbors = makelist(nlist_neighbors, nlist_max);
  //PyObject * ptr_nlist_offset = makelist(nlist_offset,nlist_max*3);
  //PyObject * arglist = Py_BuildValue("(OOOOi)", ptr, ptr_nlist_mapping, ptr_nlist_neighbors, ptr_nlist_offset, vflag);

  /*intptr_t ptr_nlist_mapping = (intptr_t) (*((void **) nlist_mapping));
  intptr_t ptr_nlist_neighbors = (intptr_t) (*((void **) nlist_neighbors));
  intptr_t ptr_nlist_offset = (intptr_t) (*((void **) nlist_offset));
  PyObject * arglist = Py_BuildValue("(Oiiii)", ptr, ptr_nlist_mapping, ptr_nlist_neighbors, ptr_nlist_offset, vflag);*/


  //PyObject * arglist = Py_BuildValue("(Oiiii)", ptr, nlist_mapping, nlist_neighbors, nlist_offset, vflag);

  PyObject * arglist = Py_BuildValue("(Oi)", ptr, vflag);

  PyObject * result = PyEval_CallObject((PyObject*)pFunc, arglist);

  //RS now extract float value from result and set it to py_energy
  //  TODO: what about virial? if vflag is True?
  if (!result) {
    PyErr_Print();
    PyErr_Clear();
    PyGILState_Release(gstate);
    error->all(FLERR,"Calling external Python energy function failed .. no energy returned");
    }
  py_energy = PyFloat_AsDouble(result);
  Py_DECREF(result);
  //RS end

  
  Py_DECREF(arglist);
  //Py_CLEAR(arglist);

  
  PyGILState_Release(gstate);
}

void FixPythonInvoke::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

// JPD this function computes the neighbors based on the LAMMPS neighborlist
// and transfers it to python. Based on the nnp pair style.
// THIS WORKS ONLY IN SERIAL CASE AND FOR ORTHORHOMBIC CELLS SO FAR
void FixPythonInvoke::transferNeighborList()
{
  neighbor->build_one(list);

  // Transfer neighbor list to Python.
  double rc2 = maxCutoffRadius * maxCutoffRadius;
  for (int ii = 0; ii < list->inum; ++ii) {
    //std::cout<< "after" << std::endl;
    int i = list->ilist[ii];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      int j = list->firstneigh[i][jj];
      j &= NEIGHMASK;
      double dx = atom->x[i][0] - atom->x[j][0];
      double dy = atom->x[i][1] - atom->x[j][1];
      double dz = atom->x[i][2] - atom->x[j][2];
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= rc2) {

        int pconn_x = 0;
        int pconn_y = 0;
        int pconn_z = 0;

        if(j<atom->natoms){
          //pconn_x = pconn_y = pconn_z = 0;
          // j is a real atom pconn = 000
          PyObject * arglist = Py_BuildValue("(iiiii)",i,j,pconn_x,pconn_y,pconn_z);
          PyEval_CallObject((PyObject*)pNeighFunc, arglist);
        } else {
       
          pconn_x = round((atom->x[j][0]-atom->x[atom->map(atom->tag[j])][0])/domain->xprd);
          pconn_y = round((atom->x[j][1]-atom->x[atom->map(atom->tag[j])][1])/domain->yprd);
          pconn_z = round((atom->x[j][2]-atom->x[atom->map(atom->tag[j])][2])/domain->zprd);

          PyObject * arglist = Py_BuildValue("(iiiii)",i,atom->map(atom->tag[j]),pconn_x,pconn_y,pconn_z);
          PyEval_CallObject((PyObject*)pNeighFunc, arglist);
        }
      }
    }
  }
}


void FixPythonInvoke::computeNeighborList()
{
  neighbor->build_one(list);
  // prepare python neighborlist
  int i_neighbor = 0;
  // following should be replaced by memset
  memset(nlist_mapping, 0, atom->natoms*sizeof(int));
  //for (int i = 0; i < atom->natoms; i++){
  //  nlist_mapping[i]=0;
  //}
  // Transfer neighbor list to Python.
  double rc2 = maxCutoffRadius * maxCutoffRadius;
  for (int ii = 0; ii < list->inum; ++ii) {
    //std::cout<< "after" << std::endl;
    int i = list->ilist[ii];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      int j = list->firstneigh[i][jj];
      j &= NEIGHMASK;
      double dx = atom->x[i][0] - atom->x[j][0];
      double dy = atom->x[i][1] - atom->x[j][1];
      double dz = atom->x[i][2] - atom->x[j][2];
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= rc2) {

        int pconn_x = 0;
        int pconn_y = 0;
        int pconn_z = 0;

        if(j>=atom->natoms){
          
          pconn_x = round((atom->x[j][0]-atom->x[atom->map(atom->tag[j])][0])/domain->xprd);
          pconn_y = round((atom->x[j][1]-atom->x[atom->map(atom->tag[j])][1])/domain->yprd);
          pconn_z = round((atom->x[j][2]-atom->x[atom->map(atom->tag[j])][2])/domain->zprd);

        }
        // write to python neighborlist
        // check first if nlist_neighbors is not already full
        if (i_neighbor == nlist_max-1){
          // we have to rebuild by first destroying the old ones
          nlist_max = nlist_max + 2*atom->natoms;
          nlist_neighbors = (int *)realloc(nlist_neighbors,sizeof(int)*nlist_max);
          nlist_offset = (int *)realloc(nlist_offset, sizeof(int)*nlist_max*3);
          std::cout << "Nlist reallocated!!!" << std::endl;
          std::cout << nlist_max << std::endl;
        }
        nlist_neighbors[i_neighbor] = atom->map(atom->tag[j]);
        nlist_offset[3*i_neighbor+0] = pconn_x;
        nlist_offset[3*i_neighbor+1] = pconn_y;
        nlist_offset[3*i_neighbor+2] = pconn_z;
        if(nlist_mapping[i]==0){
            if(i==0){
              nlist_mapping[i]=1;
            } else {
              nlist_mapping[i]=nlist_mapping[i-1]+1;
            } 
          } else {
            nlist_mapping[i]++;
          }
        i_neighbor++;
      }
    }
  }
}



/*void FixPythonInvoke::transferNeighborList()
{
  int i = 1;
  int j = 2;
  PyObject * arglist = Py_BuildValue("(ii)",i,j);
  PyEval_CallObject((PyObject*)pNeighFunc, arglist);
}*/


/* ---------------------------------------------------------------------- */
/*  RS .. add callback for minimize                                       */

void FixPythonInvoke::min_setup(int vflag)
{
  post_force(vflag);
}

void FixPythonInvoke::min_post_force(int vflag)
{
  post_force(vflag);
}

double FixPythonInvoke::compute_scalar()
{
  return py_energy;
}
