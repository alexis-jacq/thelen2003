import numpy as np

'''
# total force depend on 3 elements :

# <-- F ___/contractile---parallele--\___ F -->
#          \-----------tendon--------/

# contractile + parallele = muscle

# so far we suppose alpha = 0 (no pennation angles):
'''

# CONSTANTS
#----------

# length muscle+tendon:
lmt0 = 0.3

# muscle force-length constants:
gamma = 0.45 # muscle shape factor
kpe = 5 # normalized muscle fiber length
epsm0 = 0.6 # passive muscle strain due to maximum isometric force

# tendon force-length constants:
ltslack = 1 # normalized tendon slack length
epst0 = 0.04 # tendon strain at the maximal isometric muscle force
kttoe = 3 # linear scale factor

# muscle force-velocity constants:
lmopt  = 1 # optimal muscle fiber length
vmmax =1 # maximum muscle velocity for concentric activation
fmlen = 1.4 # normalized maximum force generated at the lengthening phase
af = 0.25 # shape factor

# muscle activation constants:
t_act = 0.015 # activation time constant [s]
t_deact = 0.05 # deactivation time constant [s]

# FORCE FUNCTIONS
#-----------------

# force from the contractile element (gaussian)
def length_contractile_force(lm):
    '''force of the contractile element as function of muscle length.
    lm = muscle lentgh'''
    fce = np.exp(-(lm-1)**2/gamma)
    return fce

# force from the parallele element (elastic when longer than slack)
def length_parallele_force(lm):
    '''force of the parallele element as function of muscle length.
    lm = muscle lentgh
    '''
    if lm < 1:
        fpe = 0
    else:
        fpe = (np.exp(kpe*(lm-1)/epsm0)-1)/(np.exp(kpe)-1)
    return fpe

# muscle force (sum of activation*contractile + parallele)
def length_muscle_force(lm, al=1):
    '''total force as function of muscle length.
    lm = muscle length
    al = activation level, between 0 and 1
    '''
    al = min(max(0,al),1)
    return length_parallele_force(lm) + al*length_contractile_force(lm)

# tendon force
def length_tendon_force(lt):
    '''force of tendon as function of tendon length.
    lt = tendon length
    '''
    epst = (lt-ltslack)/ltslack
    fttoe = 0.33
    epsttoe =  .99*epst0*np.e**3/(1.66*np.e**3 - .67)
    ktlin =  .67/(epst0 - epsttoe)
    if epst <= 0:
        fse = 0
    elif epst <= epsttoe:
        fse = fttoe/(np.exp(kttoe)-1)*(np.exp(kttoe*epst/epsttoe)-1)
    else:
        fse = ktlin*(epst-epsttoe) + fttoe
    return fse

# velocity of muscle from muscle force
def force_muscle_velocity(fm, flce=1, a=1):
    """muscle velocity as function of CE force.
    fm = normalized muscle force
    flce = normalized muscle force due to the force-length relationship
    a = muscle activation
    """

    vmmax_ = vmmax*lmopt
    if fm <= a*flce:  # isometric and concentric activation
        b = a*flce + fm/af
    else:             # eccentric activation
        b = (2 + 2/af)*(a*flce*fmlen - fm)/(fmlen - 1)
    vm = (0.25  + 0.75*a)*vmmax_*(fm - a*flce)/b

    return vm

# force of muscle from muscle velocity (inverse of force_muscle_velocity)
def velocity_muscle_force(vm, flce=1, a=1):
    """force of the contractile element as function of muscle velocity.
    vm = normalized muscle velocity
    flce = normalized muscle force due to the force-length relationship
    a = muscle activation level
    """

    vmmax_ = vmmax*lmopt
    if vm <= 0:  # isometric and concentric activation
        fvce = af*a*flce*(4*vm + vmmax_*(3*a + 1))/(-4*vm + vmmax_*af*(3*a + 1))
    else:        # eccentric activation
        fvce = a*flce*(af*vmmax_*(3*a*fmlen - 3*a + fmlen - 1) + 8*vm*fmlen*(af + 1)) /\
        (af*vmmax_*(3*a*fmlen - 3*a + fmlen - 1) + 8*vm*(af + 1))

    return fvce

# derivative of activation from excitation
def d_activation(a, u):
    '''activation dynamics, the derivative of `a` at `t`.
    a = muscle activation
    u = muscle exitaction
    '''

    if u > a:
        adot = (u - a)/(t_act*(0.5 + 1.5*a))
    else:
        adot = (u - a)/(t_deact/(0.5 + 1.5*a))

    return adot

# SIMULATOR
#----------

class Muscle():

    def __init__(sim):
        ''' input is the initial muscle lenght
        for mujoco, this is given by sim.data.actuator_length
        '''
        self.a = sim.data.actuator_length
        self.lm = sim.data.actuator_length
        self.lt = lmt0 - self.lm
        self.vm = 0

    def update_forces(self):
        ''' update forces with the current state'''
        self.fce = length_contractile_force(self.lm)
        self.fpe = length_parallele_force(self.lm)
        self.ft = length_tendon_force(self.lt)
        self.fce_t = self.ft - self.fpe

    def update(self, u):
        ''' update all muscle features from an excitation u
        the main output is the new lm,
        for mujoco, u = sim.data.ctrl
        '''
        self.update_forces()
        # velocity:
        self.vm = force_muscle_velocity(fm=self.fce_t, flce=self.fce, a=self.a)
        # muscle length:
        self.lm += self.vm
        # activation:
        d_a = d_activation(a, u)
        self.a += d_a

if __name__=="__main__":

    import matplotlib.pyplot as plt

    # test muscle force_length relationship:
    lengths = 1.6*np.arange(100)/100.
    fce = np.array([length_contractile_force(l) for l in lengths])
    fpe = np.array([length_parallele_force(l) for l in lengths])
    fm = fce + fpe
    plt.subplot(131)
    plt.plot(lengths, fce, 'g')
    plt.plot(lengths, fpe, 'r')
    plt.plot(lengths, fm, 'b')
    plt.title('muscle force-length')

    # test tendon force_length relationship:
    ft = np.array([length_tendon_force(l) for l in lengths])
    plt.subplot(132)
    plt.plot(lengths, ft)
    plt.title('tendon force-length')

    # test force_muscle_velocity:
    velocities = 1-2*np.arange(100)/100.
    fm = np.array([velocity_muscle_force(v) for v in velocities[::-1]])
    plt.subplot(133)
    plt.plot(velocities[::-1], fm)
    plt.title('muscle force-velocity')

    plt.show()
