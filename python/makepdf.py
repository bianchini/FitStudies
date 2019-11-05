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

def save_snapshot_2D(xbins, ybins, zvals, norm, title='', name=''):
  if norm:
    zvals /= zvals.sum()
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
  def __init__(self, eta_bins, pt_bins, bT_npts, bL_npts, alphaT, alphaL, algo, get, verbose) :
    self.eta_bins = eta_bins
    self.pt_bins = pt_bins
    self.bT_npts = bT_npts
    self.bL_npts = bL_npts
    self.alphaT = alphaT
    self.alphaL = alphaL
    self.lep_mass = 0.105
    self.MW = 80.000
    self.GW = 2.5
    self.MW2 = self.MW**2
    self.GW2 = self.GW**2
    self.normBW = 1.0
    self.init_roots()
    self.algo = algo
    self.get = get
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

  def ansatz_pdf(self, bT, bL, sigma=2.0):
    y = self.convert_to_y(bL)
    pdf_L = (np.exp( -0.5*y**2/sigma )/np.sqrt(2*np.pi)/sigma)*2
    pdf_T = (bT*np.exp(-bT))
    return pdf_T*pdf_L

  def convert_to_boost(self, bT, bL):
    betagammaT = self.convert_to_betagammaT(bT)
    betaL = np.tanh(self.convert_to_y(bL))
    gamma = np.sqrt((1+betagammaT**2)/(1-betaL**2))
    betagammaL = betaL*gamma
    return (betagammaT,betagammaL,gamma)

  def boost_to_CS_matrix(self, p, boost):
    betagammaT,betagammaL,gamma = boost
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
    res = norm/((x**2-self.MW2)**2 + self.MW2*self.GW2)
    return res

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

    if self.algo=='nquad':
      eta,pt,phi = x0,x1,x2
    elif self.algo in ['tplquad', 'tplquad-improved']:
      eta,pt,phi = x2,x1,x0
    elif self.algo=='fast':
      eta,pt,phi = x1,x2,x0

    if self.verbose: print("**** integrand() ****")
    if self.verbose: print("(eta,pt,phi):", eta,pt,phi, ", (bT,bL,wT,wL):", t0,t1,w0,w1)

    p = np.array([pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)])
    if self.verbose: print("p3 (lab):", p)
    boost = self.convert_to_boost( t0, t1)

    res = 0.0
    for sign in [-1,+1]:
      res_tmp = 0.0

      psCS = self.boost_to_CS_matrix(p, np.array([boost[0],sign*boost[1],boost[2]])  )
      if self.verbose: print("\t[%s] (E*,cos*,phi*):" % sign, psCS)

      res_tmp += self.BreitWignerQ(psCS[0]*2.0)/self.normBW
      if self.verbose: print("\tBreitWignerQ",res_tmp)

      res_tmp *= self.angular_pdf_CS(psCS[1], psCS[2], coeff=[0.,0.,0.,0.,0.,0.,0.,0.])
      if self.verbose: print("\tangular_pdf_CS:",res_tmp)

      res_tmp *= (4*psCS[0]/(psCS[0]**2-self.lep_mass**2))
      if self.verbose: print("\tJacobian:",res_tmp)

      res += res_tmp
      if self.verbose: print("\tres:", res)


    # Integration over phi and cos->-cos
    res *= 2*np.pi*2
    if self.verbose: print("Integration over phi:", res)

    # the quadrature weights
    if self.get == 'quadrature':
      # Jacobian of bL [a,b] --> [-1,+1]    
      res *= (self.alphaL[1]-self.alphaL[0])*0.5
      if self.verbose: print("Jacobian of bL:",res)
      res *= (w0*w1)
      if self.verbose: print("Quadrature weights:", res)
    elif self.get == 'pdf':
      res *= self.ansatz_pdf(t0,t1,sigma=2.0)      
      if self.verbose: print("Ansatz pdf (xexp factored):", res)
      res /= (t0*np.exp(-t0))

    # rhs factors
    res *= pt/2.0
    if self.verbose: print("===> integrand: :", res)
    return res
    

  def find_roots(self, eta,pt,t0,t1, nwidths, in_out):
    gammabetaT,gammabetaL,gamma = self.convert_to_boost(t0,t1)
    if gammabetaL<1e-06: 
      if in_out==0:
        if self.verbose: print("R0: [0, np.pi/3]")
        return [0, np.pi/3] 
      elif in_out==1: 
        if self.verbose: print("R1: [np.pi/3, 2*np.pi/3]")
        return [np.pi/3, 2*np.pi/3]
      else:
        if self.verbose: print("R2: [2*np.pi/3, np.pi]")
        return [2*np.pi/3, np.pi]
    betaT = gammabetaT/gamma
    betaL = gammabetaL/gamma
    E,pL = pt*np.cosh(eta),pt*np.sinh(eta)
    offset = (E - betaL*pL)/(betaT*pt)    
    x1,x2 = -0.5*(self.MW + nwidths*self.GW)/(gammabetaT*pt) + offset, -0.5*(self.MW - nwidths*self.GW)/(gammabetaT*pt) + offset
    if self.verbose: print(x1,x2)
    if x1<-1.0:
      if x2<-1.0 or x2>1.0:
        if in_out==0:
          if self.verbose: print("R3:  [0, np.pi/3]")
          return [0, np.pi/3]
        elif in_out==1:
          if self.verbose: print("R4: [np.pi/3, 2*np.pi/3]")
          return [np.pi/3, 2*np.pi/3]
        else:
          if self.verbose: print("R5: [2*np.pi/3, np.pi]")
          return [2*np.pi/3, np.pi]
      else:  
        if in_out==0:
          if self.verbose: print("R6: [np.arccos(x2), np.pi]")
          return [np.arccos(x2), np.pi] 
        elif in_out==1:
          if self.verbose: print("R7: [0, np.arccos(x2)/2]")
          return [0, np.arccos(x2)/2]
        else:
          if self.verbose: print("R8: [np.arccos(x2)/2, np.arccos(x2)]")
          return [np.arccos(x2)/2, np.arccos(x2)]
    elif x1>=-1.0 and x1<=1.0:
      if x2<=1.0:
        if in_out==0:
          if self.verbose: print("R9: [np.arccos(x2), np.arccos(x1)]")
          return [np.arccos(x2), np.arccos(x1)]
        elif in_out==1:
          if self.verbose: print("R10: [0, np.arccos(x2)]")
          return [0, np.arccos(x2)]
        else:
          if self.verbose: print("R11: [np.arccos(x1), np.pi]")
          return [np.arccos(x1), np.pi] 
    else:
       if in_out==0:
         if self.verbose: print("R12: [0, np.pi/3]")
         return [0, np.pi/3]
       elif in_out==1:
         if self.verbose: print("R13: [np.pi/3, 2*np.pi/3]")
         return [np.pi/3, 2*np.pi/3]
       else:
         if self.verbose: print("R14: [2*np.pi/3, np.pi]")
         return [2*np.pi/3, np.pi]

  def integ(self, t0, t1, w0, w1,  x0L, x0U, x1L, x1U):
    if self.algo=='nquad':
      res = integrate.nquad(self.integrand, [[x0L, x0U], [x1L, x1U], [0,+np.pi*1] ], args=(t0,t1, w0, w1))  
    elif self.algo=='tplquad':
      res = integrate.tplquad(self.integrand, x0L, x0U, lambda x0: x1L, lambda x0 : x1U, lambda x0,x1 : 0, lambda x0,x1 : np.pi, args=(t0,t1,w0,w1))
    elif self.algo=='tplquad-improved':
      res1 = integrate.tplquad(self.integrand, x0L, x0U, lambda x0: x1L, lambda x0 : x1U, 
                               lambda x0,x1 : self.find_roots(x0,x1,t0,t1,5.0, in_out=0)[0], 
                               lambda x0,x1 : self.find_roots(x0,x1,t0,t1,5.0,in_out=0)[1], args=(t0,t1,w0,w1))
      print(res1)
      res2 = integrate.tplquad(self.integrand, x0L, x0U, lambda x0: x1L, lambda x0 : x1U, 
                               lambda x0,x1 : self.find_roots(x0,x1,t0,t1,5.0, in_out=1)[0], 
                               lambda x0,x1 : self.find_roots(x0,x1,t0,t1,5.0,in_out=1)[1], args=(t0,t1,w0,w1))
      print(res2)
      res3 = integrate.tplquad(self.integrand, x0L, x0U, lambda x0: x1L, lambda x0 : x1U, 
                               lambda x0,x1 : self.find_roots(x0,x1,t0,t1,5.0, in_out=2)[0], 
                               lambda x0,x1 : self.find_roots(x0,x1,t0,t1,5.0,in_out=2)[1], args=(t0,t1,w0,w1))
      print(res3)
      res = (res1[0]+res2[0]+res3[0], res1[1]+res2[1]+res3[1])

    elif self.algo=='fast':
      res1 = integrate.quad(self.integrand, 0, np.pi, args=( (x0L+x0U)*0.5, (x1L+x1U)*0.5, t0,t1, w0, w1) )
      res = (res1[0]*(x0U-x0L)*(x1U-x1L),  res1[1]*(x0U-x0L)*(x1U-x1L))

    return res

  def integ_bin(self, eta_bin=[0.,0.], pt_bin=[0.,0.]):
    res,res_err = (np.zeros(shape=(self.bL_npts,self.bT_npts), dtype=np.float64),
                   np.zeros(shape=(self.bL_npts,self.bT_npts), dtype=np.float64))
    for iL in range(self.bL_npts):
      for iT in range(self.bT_npts):
        if iL%5==0 and iT%5==0: print("\t(bL,bT) roots (%s,%s)" % (iL,iT))
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
        if ieta%5==0 and ipt%5==0: print("Integrating (eta,pt) bin (%s,%s)" % (ieta,ipt))
        (res[ieta,ipt], res_err[ieta,ipt]) = self.integ_bin(self.eta_bins[ieta:ieta+2], self.pt_bins[ipt:ipt+2] )
    return (res, res_err)

if __name__ == '__main__': 

  #eta_bins = np.linspace(0.0, 1.0, 21)
  #pt_bins  = np.linspace(35, 50, 16)

  eta_bins = np.array([-0.05, +0.05])
  pt_bins = np.array([37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5])

  makePdf = MakePdf(eta_bins=eta_bins, pt_bins=pt_bins, bT_npts=80, bL_npts=5, alphaT=[0.05], alphaL=[0.0, 1.0], algo='fast', get='pdf', verbose=0)
  
  clock = time.time()
  #makePdf.debug_boost()
  makePdf.nornalizeBreitWigner(algo='quad', nwidths=100)
  
  (res,res_err) = makePdf.integ_all()
  clock -= time.time()
  print('Integration done in '+'{:4.3f}'.format(-clock)+' seconds')
  save_snapshot_2D(xbins=eta_bins, ybins=pt_bins, zvals=np.sum(res, axis=(2,3)), norm=1, title='test', name='test')
  save_snapshot_2D(xbins=eta_bins, ybins=pt_bins, zvals=np.sum(res_err, axis=(2,3))/np.sum(res, axis=(2,3)), norm=0, title='testerr', name='testerr')
  np.savez('pdf', res, res_err, eta_bins, pt_bins,  makePdf.convert_to_y(makePdf.bL_pts[0]), makePdf.convert_to_betagammaT(makePdf.bT_pts[0]))
