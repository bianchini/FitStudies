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


class MakePdf():
  def __init__(self, eta_bins, pt_bins, bT_npts, bL_npts, alphaT, alphaL, verbose) :
    self.eta_bins = eta_bins
    self.pt_bins = pt_bins
    self.bT_npts = bT_npts
    self.bL_npts = bL_npts
    self.alphaT = alphaT
    self.alphaL = alphaL
    self.lep_mass = 0.105
    self.MW = 80.419 
    self.GW = 2.5
    self.MW2 = self.MW**2
    self.GW2 = self.GW**2
    self.normBW = 1.0
    self.init_roots()
    self.verbose = verbose

  def init_roots(self):
    self.bT_pts = special.roots_genlaguerre(n=self.bT_npts, alpha=1.0)
    self.bL_pts = special.roots_legendre(n=self.bL_npts)        
    print("Roots for betagammaT: ", self.convert_to_betagammaT(self.bT_pts[0]))
    print("Roots for y: ", self.convert_to_y(self.bL_pts[0]))

  def convert_to_betagammaT(self, bT):
    return bT*self.alphaT[0]

  def convert_to_y(self, bL):
    return bL*(self.alphaL[1]-self.alphaL[0])*0.5 + (self.alphaL[1]+self.alphaL[0])*0.5

  def convert_to_boost(self, bT, bL):
    betagammaT = self.convert_to_betagammaT(bT)
    betaL = np.tanh(self.convert_to_y(bL))
    gamma = np.sqrt((1+betagammaT**2)/(1-betaL**2))
    betagammaL = betaL*gamma
    return (betagammaT,betagammaL,gamma)

  def boost_to_CS_matrix(self, p, boost):
    betagammaT,betagammaL,gamma = boost
    #r = R.from_rotvec(-boost_phi * np.array([0, 0, 1]))    
    #print("Before rotation", p)
    #p3_lab = r.apply(p)
    #print("After rotation", p3_lab)
    #p4_lab = np.array([np.sqrt(np.sum(p3_lab**2)+self.lep_mass**2), p3_lab[0], p3_lab[1], p3_lab[2]])    
    p4_lab = np.array([ np.sqrt(np.sum(p**2)+self.lep_mass**2), p[0], p[1], p[2] ])
    Xt = math.sqrt(1 + betagammaT**2)
    boost_matrix = np.array([[ gamma, -betagammaT, 0, -betagammaL ],
                             [ -betagammaT*gamma/Xt, Xt, 0, betagammaT*betagammaL/Xt],
                             [0., 0., 1., 0.],
                             [-betagammaL/Xt, 0, 0, gamma/Xt]
                             ] )
    p4_CS = np.linalg.multi_dot([boost_matrix, p4_lab])
    flip_z = -1 if betagammaL<0.0 else +1
    return np.array([p4_CS[0], p4_CS[3]/np.sqrt(np.sum(p4_CS[1:]**2))*flip_z, np.arctan2(p4_CS[2],p4_CS[1])*flip_z]) 

  def debug_boost(self):
    bT = 0.1
    bL = +0.9
    pt = 35.0
    pz = 0.0 
    phi = -10.
    p = np.array([pt*np.cos(np.pi/180.*phi), pt*np.sin(np.pi/180.*phi), pz])
    print("p4 lab: ", p)
    boost = self.convert_to_boost(bT,bL)
    print("boost vector: ", np.array([0, boost[0], boost[1]])/boost[2] )
    ps = self.boost_to_CS_matrix( p, boost)
    print("p4 CS : ", ps)

  def angular_pdf_CS(self, x, y, coeff=[]):
    UL = (1.0 + x*x)
    L = 0.5*(1-3*x*x)
    T = 2.0*x*np.sqrt(1-x*x)*np.cos(y)
    I = 0.5*(1-x*x)*np.cos(2*y)
    A = np.sqrt(1-x*x)*np.cos(y)
    P = x
    p7 = (1-x*x)*np.sin(2*y)
    p8 = 2.0*x*np.sqrt(1-x*x)*np.sin(y)
    p9 = np.sqrt(1-x*x)*np.sin(y)
    return 3./16./math.pi * ( UL + coeff[0]*L + coeff[1]*T + coeff[2]*I + coeff[3]*A + coeff[4]*P + coeff[5]*p7 + coeff[6]*p8 + coeff[7]*p9)

  # x: Q**2
  def BreitWignerQ2(self, x):
    return np.sqrt(x)/((x-self.MW2)**2 + x**2*self.GW2/self.MW2)

  # x: Q
  def BreitWignerQ(self, x):
    gB = np.sqrt(self.MW2*(self.MW2 + self.GW2))
    norm = 2.0*np.sqrt(2.0)*self.MW*self.GW*gB/np.pi/np.sqrt(self.MW2+gB)
    #return norm*x**2/((x**2-self.MW2)**2 + x**4*self.GW2/self.MW2)
    return norm/((x**2-self.MW2)**2 + self.MW2*self.GW2)

  def nornalizeBreitWigner(self, algo='vegas', nwidths=100):
    def Integrand(ndim, xx, ncomp, ff, userdata):
      a = self.MW - nwidths*self.GW
      if a < 0.:
        a = 0.0
      b = self.MW + nwidths*self.GW
      x = xx[0]*(b-a) + a
      result = self.BreitWignerQ(x)
      ff[0] = result*(b-a)
      return 0
    NDIM = 1 
    if algo=='chure':   
      res = pycuba.Cuhre(Integrand, NDIM, key=13, verbose=0)['results'][0] 
    elif algo=='vegas': 
      res = pycuba.Vegas(Integrand, NDIM , epsrel=0.0001, epsabs=1e-12, maxeval=1000000)['results'][0]
    elif algo=='quad':
      res = integrate.quad(self.BreitWignerQ, self.MW-nwidths*self.GW if self.MW-nwidths*self.GW>0. else 0.0, self.MW+nwidths*self.GW)[0]
    self.normBW = res

  # x0,x1,x2: eta,pt,phi-phiW
  # t0,t1: bT, bL
  # w0,w1: w_bT, w_bL    
  def integrand(self, x0,x1,x2,t0,t1,w0,w1):
    if self.verbose: print("**************")
    res = 1.0
    if self.verbose: print(x0,x1,x2,t0,t1,w0,w1)
    p = np.array([x1*np.cos(x2), x1*np.sin(x2), x1*np.sinh(x0)])
    if self.verbose: print(p)
    boost = self.convert_to_boost( t0, t1)
    psCS = self.boost_to_CS_matrix(p,boost)
    if self.verbose: print(psCS)
    res *= self.BreitWignerQ(psCS[0])/self.normBW
    if self.verbose: print(res)
    res *= self.angular_pdf_CS(psCS[1], psCS[2], coeff=[0.0]*8)
    if self.verbose: print(res)
    # Jacobian of bL [a,b] --> [-1,+1]
    res *= self.alphaL[1]*0.5
    if self.verbose: print(res)
    # Jacobian
    res *= (4*psCS[0]/(psCS[0]**2-self.lep_mass**2))
    if self.verbose: print(res)
    # Integration over phi
    res *= 2*np.pi
    if self.verbose: print(res)
    # the quadrature weights
    res *= (w0*w1)
    if self.verbose: print(res)
    # rhs factors (FIX, should be mean <1/pt>^-1)
    res *= 8.0*x0
    if self.verbose: print(res)
    return res
    
  def integ(self, t0, t1, w0, w1,  x0L, x0U, x1L, x1U):
    #res = integrate.nquad(self.integrand, [[x0L, x0U], [x1L, x1U], [-np.pi,+np.pi] ], args=(t0,t1, w0, w1))  
    res = integrate.tplquad(self.integrand, x0L, x0U, lambda x: x1L, lambda x : x1U, lambda x,y : -np.pi, lambda x,y : +np.pi, args=(t0,t1,w0,w1))
    return res

  def integ_bin(self, eta_bin=[0.,0.], pt_bin=[0.,0.]):
    res,res_err = (np.zeros(shape=(self.bL_npts,self.bT_npts), dtype=np.float64),
                   np.zeros(shape=(self.bL_npts,self.bT_npts), dtype=np.float64))
    for iL in range(self.bL_npts):
      for iT in range(self.bT_npts):
        (res[iL,iT],res_err[iL,iT]) = self.integ(self.bT_pts[0][iT], self.bL_pts[0][iL], 
                                                 self.bT_pts[1][iT], self.bL_pts[1][iL],
                                                 eta_bin[0], eta_bin[1], 
                                                 pt_bin[0], pt_bin[1])      
    return (res,res_err)

  def integ_all(self):
    res,res_err = (np.zeros(shape=(self.eta_bins.size-1, self.pt_bins.size-1, self.bL_npts,self.bT_npts), dtype=np.float64), 
                   np.zeros(shape=(self.eta_bins.size-1, self.pt_bins.size-1, self.bL_npts,self.bT_npts), dtype=np.float64))
    for ieta in range(self.eta_bins.size-1):
      for ipt in range(self.pt_bins.size-1):
        print("Integrating (eta,pt) bin (%s,%s)" % (ieta,ipt))
        (res[ieta,ipt], res_err[ieta,ipt]) = self.integ_bin(self.eta_bins[ieta:ieta+2], self.pt_bins[ipt:ipt+2] )
    return (res, res_err)

if __name__ == '__main__': 

  eta_bins = np.linspace(0, 2.5, 2)
  pt_bins  = np.linspace(25, 55, 3)

  eta_bins = np.array([0.0, 0.05])
  pt_bins = np.array([40.0, 40.5, 41.0, 42.0])

  makePdf = MakePdf(eta_bins=eta_bins, pt_bins=pt_bins, bT_npts=1, bL_npts=1, alphaT=[0.0], alphaL=[0., 0.01], verbose=0)
  
  clock = time.time()
  makePdf.debug_boost()
  makePdf.nornalizeBreitWigner(algo='quad', nwidths=100)
  
  (res,res_err) = makePdf.integ_all()
  clock -= time.time()
  print('Integration done in '+'{:4.3f}'.format(-clock)+' seconds')
  save_snapshot_2D(xbins=eta_bins, ybins=pt_bins, zvals=np.sum(res, axis=(2,3)), title='test', name='test')
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

