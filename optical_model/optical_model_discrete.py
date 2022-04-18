#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class optical_model:
    def __init__(self, matched_quad_settings, beamEnergyMeV, gdet_max = 1., emitx_u=0.4e-6, emity_u=0.4e-6):
        
        #unpack arguments
        
        self.matched_quad_settings = np.copy(matched_quad_settings) #1d array of length 16 containing field integrals for each ltu quad starting at 620
        self.beamEnergyMeV = np.copy(beamEnergyMeV) #beam energy in MeV
        self.gdet_max = np.copy(gdet_max)
        self.emitx_u = np.copy(emitx_u) #unnormalized emittance in x
        self.emity_u = np.copy(emity_u) #unnormalized emittance in y
        self.drifts = np.array([ 4.15, 12.15,  0.78,  3.26, 10.84, 17.52, 17.52, 17.52, 17.53, 17.53, 14.58, 6.39,
  8.14,  4.14, 10.9,   0.  ]) # the drift in meters following each ltu quad (taken from lattice geometry) harcoded because they never change
        
        #calc the matched twiss params (beta) for the smooth focusing model of the undulator
        betaxf, betayf = self.calcmatch_FODO()
#         self.smooth_beta_match = 0.5*(betaxf + betayf)
#         self.txf_design = np.array([self.smooth_beta_match,0.])
#         self.tyf_design = np.array([self.smooth_beta_match,0.])
        self.txf_design = np.array([betaxf,0.])
        self.tyf_design = np.array([betayf,0.])        
        self.nq = len(self.matched_quad_settings) #number of quads
        
        #store matched settings as the current settings and calculate current cumulative transport matrices for x and y
        self.update_settings(self.matched_quad_settings)
        
        # Backprop the matched twiss to set upstream twiss
        betaxi, alphaxi = self.back_prop(self.qx, betaxf)
        betayi, alphayi = self.back_prop(self.qy, betayf)
        self.txi = np.array([betaxi,alphaxi])
        self.tyi = np.array([betayi,alphayi])
        self.matched_beamsize = self.beamsize()[0]
        
        self.n = 4.5
    
    def update_settings(self, new_settings):
        self.quad_settings = np.copy(new_settings) #store current settings
        self.qx = self.MakeFakeQ(self.quad_settings) #cumulative transport matrix for x twiss
        self.qy = self.MakeFakeQ(-1*self.quad_settings) #cumulative transport matrix for y twiss

    def update_settings4d(self, new_settings4d):
        #update only ltu quads 620, 640, 660, 680 using a 1d vector with only 4 values (in the stated order)
        self.quad_settings[0:3] = new_settings4d[0:3]
        self.quad_settings[4] = new_settings4d[3] #skip the 3rd index in quad_settings because quad ltu 665 at index 3 is typically not tuned
        self.update_settings(self.quad_settings)
        
    def reset_to_match(self):
        self.update_settings(self.matched_quad_settings)
        
    def calcmatch_FODO(self, undQuadGrad = 37.5, quad_len = 0.08, drift_len = 3.79):
    #calculate the design twiss parameters for an undulator with the geometry specified in the arguments here.
    #note that for genesis simulation (where the undulator magnets are 30 cm), undQuadGrad = gradient integral / length = (3 T) / (0.3 m) = 10 T/m
    # for the actual accelerator lattice, the undQuadGrad = gradient integral / length = (3 T) / (0.08 m) = 37.5 T/m

        a = 3.e8/(self.beamEnergyMeV*1.e6) # electron charge divided by relativistic momentum, SI units m/(V*s)
        k = a*undQuadGrad
        l = quad_len #quad length
        L = drift_len #drift length
        MF = np.array([  [np.cos(0.5*np.sqrt(k)*l),1/np.sqrt(k)*np.sin(0.5*np.sqrt(k)*l)],
                            [-np.sqrt(k)*np.sin(0.5*np.sqrt(k)*l),np.cos(0.5*np.sqrt(k)*l)]  ])
        MD = np.array([  [np.cosh(0.5*np.sqrt(k)*l),1/np.sqrt(k)*np.sinh(0.5*np.sqrt(k)*l)],
                            [np.sqrt(k)*np.sinh(0.5*np.sqrt(k)*l),np.cosh(0.5*np.sqrt(k)*l)]  ])

        MO = np.array([[1.,L], [0,1.]])
        M1 = MF.dot(MO).dot(MD).dot(MD).dot(MO).dot(MF)

        betaMAX = M1[0,1]/np.sqrt(1-(M1[0,0])**2)

        M2 = MD.dot(MO).dot(MF).dot(MF).dot(MO).dot(MD)

        betaMIN = M2[0,1]/np.sqrt(1-(M2[0,0])**2)

        betax0 = betaMIN
        betay0 = betaMAX

        return betax0, betay0
    
    def back_prop(self, q, betaf):
    # solves for the input twiss parameters at the beginning of the lattice that, when forward propagated through the lattice, will result in the matched twiss params half through the first undulator quad.
#         betaf = self.smooth_beta_match
        alphaf = 0
        gammaf=1./betaf
        qinv = np.linalg.inv(q)
        betai, alphai, gammai = qinv.dot([betaf, alphaf, gammaf])
        return betai, alphai
    
    def MakeFakeQ(self, K):
        #compute cumulative transport matrix
        a = 3.e8/(self.beamEnergyMeV*1.e6)*0.1
        K=a*K
        assert(K.shape[0]==self.drifts.shape[0])
        q=np.eye(3)
        m = K.shape[0]
        for j in range(m):
            # add quad
            qj = np.eye(3)
            Cp=K[j]   # C=1, Sp=1, S=0 for quad
            qj[1,0] = -Cp
            qj[2,0] = Cp**2
            qj[2,1] = -2*Cp
            q = np.matmul(qj,q)

            # add drift
            qj = np.eye(3)
            S=self.drifts[j]  # C=1, Sp=1, Cp=0 for drift
            qj[0,1] = -2*S
            qj[0,2] = S**2
            qj[1,2] = -S
            q = np.matmul(qj,q)

        return q
    
    def MakeFakeQ_single(self, K, drift):
        #compute transport matrix for single quad plus subsequent drift
        a = 3.e8/(self.beamEnergyMeV*1.e6)*0.1
        K=a*K
        q=np.eye(3)
        # add quad
        qj = np.eye(3)
        Cp=K   # C=1, Sp=1, S=0 for quad
        qj[1,0] = -Cp
        qj[2,0] = Cp**2
        qj[2,1] = -2*Cp
        q = np.matmul(qj,q)

        # add drift
        qj = np.eye(3)
        S=drift  # C=1, Sp=1, Cp=0 for drift
        qj[0,1] = -2*S
        qj[0,2] = S**2
        qj[1,2] = -S
        q = np.matmul(qj,q)

        return q
    
    def TransportTwiss(self, q, t0):


        # add gamma to twiss
        g0 = (1+t0[1]**2)/t0[0]

        if len(q.shape)==3:
            nex=q.shape[0]

            # transport twiss
            tfinal=np.ones(nex,2)*t0
            tfinal[:,0] = q[:,0,0]*t0[0] + q[:,0,1]*t0[1] + q[:,0,2]*g0
            tfinal[:,1] = q[:,1,0]*t0[0] + q[:,1,1]*t0[1] + q[:,1,2]*g0

        else:
            # transport twiss
            tfinal=np.ones(2)*t0
            tfinal[0] = q[0,0]*t0[0] + q[0,1]*t0[1] + q[0,2]*g0
            tfinal[1] = q[1,0]*t0[0] + q[1,1]*t0[1] + q[1,2]*g0

        return(tfinal)
    
    def Twiss_at_wire735(self):
        qx = self.MakeFakeQ_single(self.quad_settings[0], self.drifts[0])
        qy = self.MakeFakeQ_single(-1*self.quad_settings[0], self.drifts[0])
        for i in range(1,6):
            qxnext = self.MakeFakeQ_single(self.quad_settings[i], self.drifts[i])
            qynext = self.MakeFakeQ_single(-1*self.quad_settings[i], self.drifts[i])
            qx = np.matmul(qxnext, qx)
            qy = np.matmul(qynext, qy)
        qxnext = self.MakeFakeQ_single(self.quad_settings[6], 8.67)
        qynext = self.MakeFakeQ_single(-1*self.quad_settings[6], 8.67)
        qx = np.matmul(qxnext, qx)
        qy = np.matmul(qynext, qy)
        txf = self.TransportTwiss(qx, self.txi)
        tyf = self.TransportTwiss(qy, self.tyi)
        return txf, tyf
    
    def beamsize(self):
        #computes ~beamsize 
        emitx_u = self.emitx_u
        emity_u = self.emity_u
        txf = self.TransportTwiss(self.qx, self.txi)
        tyf = self.TransportTwiss(self.qy, self.tyi)
        qf_half = self.MakeFakeQ_single(-15., 3.79)
        qd_half = self.MakeFakeQ_single(15., 3.79)
        qf = self.MakeFakeQ_single(-30., 3.79)
        qd = self.MakeFakeQ_single(30., 3.79)
        qxs = [qd_half]
        qys = [qf_half]
        for i in range(10):
            qxs += [np.matmul(qf,qxs[-1])]
            qxs += [np.matmul(qd,qxs[-1])]
            qys += [np.matmul(qd,qys[-1])]
            qys += [np.matmul(qf,qys[-1])]
        
        txs = [self.TransportTwiss(qx, txf) for qx in qxs]
        tys = [self.TransportTwiss(qy, tyf) for qy in qys]
        
        betaxs = [txf[0]]
        betays = [tyf[0]]
        
        betaxs += [tx[0] for tx in txs]
        betays += [ty[0] for ty in tys]
        
        avg_beta_x = np.mean(betaxs)
        avg_beta_y = np.mean(betays)
        
        avg_squared_size = (avg_beta_x + avg_beta_y)
        avg_size = np.sqrt(avg_squared_size)   
        
        

        return avg_size, betaxs, betays        #~rms size


    def gas_detector(self):
        #simple 1d exponential gain model
        #from Huang pgs 100-101:
        #Lg ~ rho^-1 (the gain length)
        #Lsat ~ rho^-1 (the saturation length)
        #P_in ~ rho/N_lcoh ~ rho**2     (?)
        #P(z) = P_in*exp(z/Lg)   #power as a function of z, exponential gain
        #Psat = P_in*exp(Lsat/Lg) ~ P_in ~ rho**2 (?)     power at saturation


        #beam radius, assumes equal emittance in x and y: beamsize = np.sqrt(sigma_x^2+sigma_y^2) ~ np.sqrt(beta_x + beta_y)
#         beamsize_min = np.sqrt(2*self.smooth_beta_match) 
        beamsize_min = self.matched_beamsize 
        rho_max = beamsize_min**(-2./3.) # ideal pierce parameter (up to constant of proportionality, assuming constant emittance/no Bmag)

        
        beamsize = self.beamsize()[0]
        rho = beamsize**(-2./3.) # pierce parameter (up to constant of proportionality, assuming constant emittance/no Bmag)

        
#         attenuation = (rho/rho_max)**1.*np.exp(18.*(1./rho_max)*(rho-rho_max)) #P/P_max after 18 ideal gain lengths

        attenuation = (rho/rho_max)**self.n #Psat/Psat_max

#         attenuation = (rho/rho_max)**(3./2) #Psat/Psat_max

        result = self.gdet_max*attenuation #the approximate gas detector reading at saturation
        
        return result, beamsize
    
    def gas_detector_with_noise(self, snr = 0.05, bg_std_dev = 0.1):
        #snr is the signal to noise ratio: signal_std_dev = snr*signal
        #bg_std_dev is the background noise std_dev

        gdet_noiseless = self.gas_detector()[0]

        signal_std_dev = snr*gdet_noiseless #compute std dev of the gas detector signal

        #add signal noise and background noise to the "true" gas detector value
        gdet_with_noise = gdet_noiseless + np.random.normal(scale=signal_std_dev) + np.random.normal(scale=bg_std_dev) 

        return gdet_with_noise