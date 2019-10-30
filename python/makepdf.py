import pycuba
from pprint import pprint
import scipy
from scipy import integrate, special
from scipy.spatial.transform import Rotation as R

import numpy as np
import math
import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from array import array;
import copy

from itertools import product

def save_snapshot_2D(xbins, ybins, zvals, title='', name=''):
  plt.pcolormesh(xbins, ybins, zvals.T)
  if zvals[zvals<=0.].size > 0:
    plt.colorbar()
  else:
    plt.colorbar(format='%.0e')
  plt.axis([xbins.min(), xbins.max(), ybins.min(), ybins.max()])
  plt.title(title)
  #plt.show()
  plt.savefig(name+'.png')
  plt.close('all')


eta_bins = np.array([0.0,   0.1,  0.2]) 
pt_bins  = np.array([30.0, 31.0, 32.0])
fvals = np.array([[0.1,0.2], [0.08,0.16]]) 
save_snapshot_2D(xbins=eta_bins, ybins=pt_bins, zvals=fvals, title='test', name='test')

class MakePdf():
  def __init__(self, eta_bins, pt_bins, fvals, bT_npts, bL_npts, alphaT, alphaL) :
    self.eta_bins = eta_bins
    self.pt_bins = pt_bins
    self.fvals = fvals
    self.bT_npts = bT_npts
    self.bL_npts = bL_npts
    self.bT_pts = special.roots_genlaguerre(n=bT_npts, alpha=1.0)
    self.bL_pts = special.roots_legendre(n=bL_npts)    
    self.alphaT = alphaT
    self.alphaL = alphaL
    self.lep_mass = 0.105

    #self.beta_pts = np.array(list(product(self.bT_pts[0], self.bL_pts[0])))

  def convert_to_betagamma(self, bT, bL, phiW):
    betagammaT = bT/self.alphaT[0] 
    betagammaL = np.tanh(bL*(self.alphaL[1]-self.alphaL[0])*0.5 + (self.alphaL[1]+self.alphaL[0])*0.5)
    gamma = np.sqrt(1 + betagammaT**2 + betagammaL**2)
    return np.array([betagammaT*np.cos(phiW), betagammaT*np.sin(phiW), betagammaL, gamma ])

  def boost_to_CS_matrix(self, p, boost):

    gamma = boost[-1]
    betagammaT = np.sqrt(boost[0]**2 + boost[1]**2)
    betagammaL = boost[2]
    #print("betagamma= %s %s" % (betagammaT,betagammaL))
    boost_phi = np.arctan2(boost[1],boost[0])
    #print("boost_phi", boost_phi)
    r = R.from_rotvec(-boost_phi * np.array([0, 0, 1]))    

    #print("Before rotation", p)
    p3_lab = r.apply(p)
    #print("After rotation", p3_lab)
    p4_lab = np.array([np.sqrt(np.sum(p3_lab**2)+self.lep_mass**2), p3_lab[0], p3_lab[1], p3_lab[2]])
    Xt = math.sqrt(1 + betagammaT**2)
    boost_matrix = np.array([[ gamma, -betagammaT, 0, -betagammaL ],
                      [ -betagammaT*gamma/Xt, Xt, 0, betagammaT*betagammaL/Xt],
                      [0., 0., 1., 0.],
                      [-betagammaL/Xt, 0, 0, gamma/Xt]
                      ] )
    p4_CS = np.linalg.multi_dot([boost_matrix, p4_lab])
    flip_z = -1 if betagammaL<0.0 else +1
    ps = np.array([p4_CS[0], p4_CS[3]/np.sqrt(np.sum(p4_CS[1:3]**2))*flip_z, np.arctan2(p4_CS[2],p4_CS[1])*flip_z]) 
    return ps

  def debug_boost(self):
    bT = 0.01
    bL = +0.1
    pt = 35.0
    pz = 0.0 
    phiW = 50.0
    phi = 30.
    p = np.array([pt*np.cos(np.pi/180.*phi), pt*np.sin(np.pi/180.*phi), pz])
    print("p4 lab: ", p)
    ps = self.boost_to_CS_matrix( p, self.convert_to_betagamma( bT, bL, np.pi/180.*phiW))
    print("p4 CS : ", ps)

  # x0: pt, x1: eta, x2: phi, x3: phiW
  # t0, t1: (bT, bL)
  # w0, w1: (w bT, w bL)    
  def integrand(self, x0,x1,x2,x3,t0,t1,w0,w1):
      return 1.0
    
  def integ(self, t0, t1, w0, w1,  x0L, x0U, x1L, x1U):
    res = integrate.nquad(self.integrand, [[x0L, x0U], [x1L, x1U], [-math.pi,+math.pi], [-math.pi,+math.pi]], args=(t0,t1, w0, w1))  
    return res

  def integ_bin(self, eta_bin=[0.,0.], pt_bin=[0.,0.]):
    res,res_err = (np.zeros(shape=(self.bL_npts,self.bT_npts), dtype=np.float64),
                   np.zeros(shape=(self.bL_npts,self.bT_npts), dtype=np.float64))
    for iL in range(self.bL_npts):
      for iT in range(self.bT_npts):
        (res[iL,iT],res_err[iL,iT]) = self.integ(self.bT_pts[0][iT], self.bL_pts[0][iL], 
                                                 self.bT_pts[1][iT], self.bL_pts[1][iL],
                                                 eta_bin[0], eta_bin[1], pt_bin[0], pt_bin[1])      
    return (res,res_err)

  def integ_all(self):
    res,res_err = (np.zeros(shape=(self.eta_bins.size-1, self.pt_bins.size-1, self.bL_npts,self.bT_npts), dtype=np.float64), 
                   np.zeros(shape=(self.eta_bins.size-1, self.pt_bins.size-1, self.bL_npts,self.bT_npts), dtype=np.float64))
    for ieta in range(self.eta_bins.size-1):
      for ipt in range(self.pt_bins.size-1):
        (res[ieta,ipt], res_err[ieta,ipt]) = self.integ_bin(self.eta_bins[ieta:ieta+2], self.pt_bins[ipt:ipt+2] )
    return (res, res_err)

if __name__ == '__main__': 
  makePdf = MakePdf(eta_bins=eta_bins, pt_bins=pt_bins, fvals=fvals, bT_npts=2, bL_npts=1, alphaT=[1.0], alphaL=[-3.5, 3.5])
  clock = time.time()
  makePdf.debug_boost()

  (res,res_err) = makePdf.integ_all()
  clock -= time.time()
  print('Integration done in '+'{:4.3f}'.format(-clock)+' seconds')
  pprint(res)
  pprint(res_err)
               

'''
if __name__ == '__main__':
  import math

  def Integrand(ndim, xx, ncomp, ff, userdata):
    x,y,z = [xx[i] for i in range(ndim.contents.value)]
    result = math.sin(x)*math.cos(y)*math.exp(z)
    ff[0] = result
    return 0

  func = lambda x,y,z: np.sin(x)*np.cos(y)*np.exp(z)

  NDIM = 3
  MAXEVAL = 100000
 
  from os import environ as env
  verbose = 2
  if 'CUBAVERBOSE' in env:
    verbose = int(env['CUBAVERBOSE'])

  res = pycuba.Vegas(Integrand, NDIM, verbose=0, epsrel=0.00001, epsabs=1e-12, maxeval=100000)
  print(res['results'][0])
  
  res = pycuba.Cuhre(Integrand, NDIM, key=13, verbose=0)
  print(res['results'][0])

  res = integrate.nquad(func, [[0., 1.], [0., 1.], [0.,1.]])
  print res
                        
'''

