import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import linregress as lr

class linear_roughness(object):
    """
    Processes an image in order to measure its roughness exponent
    calculating "local width", "displacement-displacement" or the
    "structure factor".
    - Initialize object with the filename: wall01 = linear_roughness('image01.tif').
    - Select region of interest and crop with wall01.crop_image().
    - Rotate as necessary with wall01.rotate_image(angle), where angle is in degrees.
    - Rotation and cropping can be performed in any order.
    - To smoothen out an impurity or defect in image use wall01.fixtomax() and click on defect. Repeat as necessary.
    - Apply gaussian filter with wall01.gaussian_image().
    - After gaussina filter compute wall with wall01.get_wall().
    - Global width: wall01.get_global_width()
    - Local width: wall01.get_local_width(). Click on left and right limits of region to calculate slope.
    - B(r): wall01.get_B_r(). Click on left and right limits of region to calculate slope.
    - S(q): wall01.get_structure_factor(). Click on left and right limits of region to calculate slope.
    - To recalculate slope, close graph and use wall01.get_zeta_lw(), wall01.get_zeta_B_r(), or wall01.get_zeta_S_q().
    - If slope is already calculated but graph was closed and you want to open it again, use wall01.get_zeta_lw(False).
    - To save all data calculated use wall01.save_data(dataname), where dataname does not have an extension.
    - To start over use wall01.reset().

    """
    def __init__(self, filename, save_folder = '', umperpix=0.117):
        self.filename = filename
        self.save_folder = save_folder
        self.umperpix = umperpix
        self.original = imageio.imread(self.filename)
        #showing original figure
        self.fig_original, self.ax_original = plt.subplots(1,1)
        self.ax_original.imshow(self.original, cmap=plt.cm.gray)
        self._rotate = np.array([])
        self._crop = np.array([])
        self._gaussian = np.array([])
        self._xfix = 0
        self._yfix = 0
        self.last_mod = 0 #0:OG, 1:crop, 2:rotate, 3:gaussian
        self.cidfix = 0
        self.cidslope = 0
        self.radius = 25
        self.threshold = 0
        self.peakdir = 0 #0:down, 1:up
        self.reach = 0
        self.angle = 0.
        self.box = []
        self.wall = np.array([])
        self.u = np.array([])
        self.z = np.array([])
        self.global_width = 0.0
        self.local_width = np.array([])
        self.rlw = np.array([])
        self.zeta_lw = 0.0
        self.zeta_lw_err = 0.0
        self.A_lw = 0.0
        self.lw_x = np.array([])
        self.rkw = np.array([])
        self.kolton_width = np.array([])
        self._kwfit = []
        self.kw_x = np.array([])
        self.zeta_kw = 0.
        self.zeta_kw_err = 0.
        self.A_kw = 0.
        self.B_r = np.array([])
        self.rB_r = np.array([])
        self.zeta_B_r = 0.0
        self.zeta_B_r_err = 0.0
        self.A_B_r = 0.0
        self.Br_x = np.array([])
        self.S_q = np.array([])
        self.q = np.array([])
        self.zeta_S_q = 0.0
        self.zeta_S_q_err = 0.0
        self.A_S_q = 0.0
        self.Sq_x = np.array([])
        self.total_clicks = 0
        self._lwfit = []
        self._brfit = []
        self._sqfit = []
        self.testarr = np.array([])
    #
    def reset(self):
        plt.close('all')
        self.fig_original, self.ax_original = plt.subplots(1,1)
        self.ax_original.imshow(self.original, cmap=plt.cm.gray)
        self._rotate = np.array([])
        self._crop = np.array([])
        self._gaussian = np.array([])
        self._xfix = 0
        self._yfix = 0
        self.last_mod = 0 #0:OG, 1:crop, 2:rotate, 3:gaussian
        self.cidfix = 0
        self.cidslope = 0
        self.radius = 25
        self.threshold = 0
        self.peakdir = 0 #0:down, 1:up
        self.reach = 0
        self.angle = 0.
        self.box = []
        self.wall = np.array([])
        self.u = np.array([])
        self.z = np.array([])
        self.global_width = 0.0
        self.local_width = np.array([])
        self.rlw = np.array([])
        self.zeta_lw = 0
        self.zeta_lw_err = 0
        self.A_lw = 0
        self.lw_x = np.array([])
        self.rkw = np.array([])
        self.kolton_width = np.array([])
        self._kwfit = []
        self.kw_x = np.array([])
        self.zeta_kw = 0.
        self.zeta_kw_err = 0.
        self.A_kw = 0.
        self.B_r = np.array([])
        self.rB_r = np.array([])
        self.zeta_B_r = 0
        self.zeta_B_r_err = 0
        self.A_B_r = 0
        self.Br_x = np.array([])
        self.S_q = np.array([])
        self.q = np.array([])
        self.zeta_S_q = 0
        self.zeta_S_q_err = 0
        self.A_S_q = 0
        self.Sq_x = np.array([])
        self.total_clicks = 0
        self._lwfit = []
        self._brfit = []
        self._sqfit = []
        self.testarr = np.array([])
    #
    def crop_image(self, box = ''):
        """
        From original image.
        Limits are rounded with int(), which may be subject to change.
        Prints out the "box" it used to crop, in the format [xleft, xright, yupper, ylower], this can
        also be used as an argument to crop to a specific box in the case, for example, of a series of
        images.
        """
        if box == '':
            xleft = int(self.ax_original.get_xlim()[0])
            xright = int(self.ax_original.get_xlim()[1])
            yupper = int(self.ax_original.get_ylim()[1])
            ylower = int(self.ax_original.get_ylim()[0])
        else:
            xleft, xright, yupper, ylower = box
        if np.array_equal(self._rotate, np.array([])):
            self._crop = self.original[yupper:ylower, xleft:xright]
        else:
            self._crop = self._rotate[yupper:ylower, xleft:xright]
        self.ax_original.clear()
        self.ax_original.imshow(self._crop, cmap=plt.cm.gray)
        self.last_mod = 1
        self.box = box
        print 'crop box = [%i, %i, %i, %i]' % (xleft, xright, yupper, ylower)
    #
    def rotate_image(self, angle):
        self.angle = angle
        if np.array_equal(self._crop, np.array([])):
            self._rotate = ndi.rotate(self.original, angle)
        else:
            self._rotate = ndi.rotate(self._crop, angle)
        self.ax_original.clear()
        self.ax_original.imshow(self._rotate, cmap=plt.cm.gray)
        self.last_mod = 2
    #
    def onclick_fix(self, event):
        if event.inaxes is not None:
            self._xfix = int(event.xdata)
            self._yfix = int(event.ydata)
        #x:column and y:row
        xy = self._xfix
        x = self._yfix
        y = xy
        print "coords (testing) = [%i, %i]" % (x,y)
        self.fig_original.canvas.mpl_disconnect(self.cidfix)
        if self.last_mod == 0:
            self.val_avg(self.original, x, y, self.radius)
            self.ax_original.clear()
            self.ax_original.imshow(self.original, cmap=plt.cm.gray)
        elif self.last_mod == 1:
            self.val_avg(self._crop, x, y, self.radius)
            self.ax_original.clear()
            self.ax_original.imshow(self._crop, cmap=plt.cm.gray)
        elif self.last_mod == 2:
            self.val_avg(self._rotate, x, y, self.radius)
            self.ax_original.clear()
            self.ax_original.imshow(self._rotate, cmap=plt.cm.gray)
        elif self.last_mod == 3:
            self.val_avg(self._gaussian, x, y, self.radius)
            self.ax_original.clear()
            self.ax_original.imshow(self._gaussian, cmap=plt.cm.gray)
        plt.draw()
    #
    def fixtomax(self, coords = ''):
        if coords == '':
            self.cidfix = self.fig_original.canvas.mpl_connect('button_press_event', self.onclick_fix)
        else:
            #x:column and y:row
            y = coords[0]
            x = coords[1]
            if self.last_mod == 0:
                self.val_avg(self.original, x, y, self.radius)
                self.ax_original.clear()
                self.ax_original.imshow(self.original, cmap=plt.cm.gray)
            elif self.last_mod == 1:
                self.val_avg(self._crop, x, y, self.radius)
                self.ax_original.clear()
                self.ax_original.imshow(self._crop, cmap=plt.cm.gray)
            elif self.last_mod == 2:
                self.val_avg(self._rotate, x, y, self.radius)
                self.ax_original.clear()
                self.ax_original.imshow(self._rotate, cmap=plt.cm.gray)
            elif self.last_mod == 3:
                self.val_avg(self._gaussian, x, y, self.radius)
                self.ax_original.clear()
                self.ax_original.imshow(self._gaussian, cmap=plt.cm.gray)
            plt.draw()
    #
    def val_avg(self, matrix, x, y, r):
        avg = (matrix[x,y-r/2] + matrix[x,y+r/2] + matrix[x-r/2,y] + matrix[x+r/2,y])/4.
        for i in range(-r/2, r/2):
            for j in range(-r/2, r/2):
                matrix[x+i,y+j] = avg
    #
    def gaussian_image(self, sigma = 2):
        if self.last_mod == 0:
            self._gaussian = ndi.gaussian_filter(self.original, sigma)
            self.ax_original.clear()
            self.ax_original.imshow(self._gaussian, cmap=plt.cm.gray)
        elif self.last_mod == 1:
            self._gaussian = ndi.gaussian_filter(self._crop, sigma)
            self.ax_original.clear()
            self.ax_original.imshow(self._gaussian, cmap=plt.cm.gray)
        elif self.last_mod == 2:
            self._gaussian = ndi.gaussian_filter(self._rotate, sigma)
            self.ax_original.clear()
            self.ax_original.imshow(self._gaussian, cmap=plt.cm.gray)
        self.last_mod = 3
    #
    def derivadatrozada(self, array, reach=20):
        self.reach = reach
        ms = []
        w = len(array)
        for i in xrange(w):
            lower = i - reach
            upper = i + reach
            if lower<0:
                lower = 0
                #upper = 2 * reach
                m, _ = np.polyfit(np.arange(lower, upper), array[lower:upper], 1)
                ms.append(m)
            elif upper>w:
                #lower = w - 2*reach
                upper = w
                m, _ = np.polyfit(np.arange(lower, upper), array[lower:upper], 1)
                ms.append(m)
            else:
                m, _ = np.polyfit(np.arange(lower, upper), array[lower:upper], 1)
                ms.append(m)
        return np.array(ms)
    #
    def get_wall(self, threshold=0.1, peakdir=0):
        #peakdir: peak direction, 0 if down, 1 if up
        self.threshold = threshold
        self.peakdir = peakdir
        if self.last_mod == 0:
            imagearray = self.original
        elif self.last_mod == 1:
            imagearray = self._crop
        elif self.last_mod == 2:
            imagearray = self._rotate
        elif self.last_mod == 3:
            imagearray = self._gaussian
        wallarray = []
        for row in imagearray:
            ms = self.derivadatrozada(row)
            msmax = ms.max()
            msmin = ms.min()
            rang = msmax - msmin
            if not peakdir:
                bottom_threshold = msmin + rang*threshold
                posts = np.arange(len(row))[ms<(bottom_threshold)]
            else:
                top_threshold = msmax - rang*threshold
                posts = np.arange(len(row))[ms>(top_threshold)]
            wallarray.append(posts.mean())
        #self.testarr = np.array(ms)
        self.wall = np.array(wallarray)
        self.ax_original.plot(self.wall, np.arange(len(self.wall)), '-r')
        self.u = self.wall * self.umperpix
        self.z = np.arange(len(self.wall)) * self.umperpix
    #
    def get_global_width(self):
        """all are numpy arrays"""
        self.global_width = self.window_averaging(self.u)
        print self.global_width
    #
    def window_averaging(self, u):
        """
        Discretizing continuous formulas, z-differential becomes
        a constant value and in this case it's the length/pixes ratio,
        d=0.117 um to a pixel. But since this is constant and the
        length of the line is taken to be L=Nd (should it be L=(N-1)d???)
        these eventually cancel out.
        Therefore we don't need to use the z array.
        """
        L = len(u)
        return (u**2).sum()/L - (u.mean())**2
    #
    def get_local_width(self):
        """
        possible_window_sizes is the 'r' variable to graph
        Must be multiplied by z[1] to get result in um at the end. At
        the beginning we keep the r array as ints so we can use them as
        indices for the u array.
        Since this calculation is not dependent on z array, it is not
        to be loaded.
        """
        N = len(self.u) #arange(1,N) stops at N-1
        possible_window_sizes = np.arange(1,N)
        w2local = []
        for windowSize in possible_window_sizes:
            parts = int(N/windowSize)
            w2local_per_window = 0
            for n in xrange(parts):
                a = self.window_averaging(self.u[(n)*windowSize:(n+1)*windowSize])
                w2local_per_window += a
            w2local.append(w2local_per_window / parts)
        self.local_width = np.array(w2local)
        self.rlw = possible_window_sizes*self.umperpix
        self.get_zeta_lw()
    #
    def get_zeta_lw(self, get_exponent=True, limits = []):
        """
        draws graph and accepts clicks to find range of slope.
        get_exponent=True brings calculated exponent.
        limits=[a,b] calculates exponent from those limits, given as integers
        corresponding to elements in x-array.
        """
        self.fig_lw, self.ax_lw = plt.subplots(1,1)
        self.ax_lw.plot(self.rlw[1:], self.local_width[1:], 'o-')
        plt.xscale('log')
        plt.yscale('log')
        self.total_clicks = 0
        self._lwfit = []
        if get_exponent and limits==[]:
            self.cidslope = self.fig_lw.canvas.mpl_connect('button_press_event', self.onclick_lw)
        elif not get_exponent and limits==[]:
            self.ax_lw.plot(self.lw_x, 10**(self.A_lw) * self.lw_x**(2*self.zeta_lw), '-', label='$\zeta_{lw}$ = %0.2f $\pm$ %0.2f' % (self.zeta_lw, self.zeta_lw_err))
            self.ax_lw.plot([],[], label="$A_{lw} = %0.2f$" % (self.A_lw))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$w^2(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
        elif limits != []:
            ini, fin = limits
            self._lwfit = limits
            p, cov = np.polyfit( np.log10(self.rlw[ini:fin]), np.log10(self.local_width[ini:fin]), 1, cov=True)
            self.lw_x = np.logspace(np.log10(self.rlw[ini]), np.log10(self.rlw[-1]), 11)
            self.zeta_lw = p[0]/2
            self.zeta_lw_err = np.sqrt(cov[0,0])/2
            self.A_lw = 10**p[1]
            self.ax_lw.plot(self.lw_x, 10**p[1] * self.lw_x**p[0], '-', label='$\zeta_{lw}$ = %0.2f $\pm$ %0.2f' % (p[0]/2, self.zeta_lw_err/2))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$w^2(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
    #
    def onclick_lw(self, event):
        if event.inaxes is not None:
            self._lwfit.append(event.xdata)
            self.total_clicks += 1
        if self.total_clicks == 2:
            self.fig_lw.canvas.mpl_disconnect(self.cidslope)
            ini = np.abs(self.rlw - self._lwfit[0]).argmin()
            fin = np.abs(self.rlw - self._lwfit[1]).argmin()
            self._lwfit[0], self._lwfit[1] = ini, fin
            print 'ini-fin = %i-%i' % (ini,fin)
            p, cov = np.polyfit( np.log10(self.rlw[ini:fin]), np.log10(self.local_width[ini:fin]), 1, cov=True)
            self.lw_x = np.logspace(np.log10(self.rlw[ini]), np.log10(self.rlw[-1]), 11)
            self.zeta_lw = p[0]/2
            self.zeta_lw_err = np.sqrt(cov[0,0])/2
            self.A_lw = p[1]
            self.ax_lw.plot(self.lw_x, 10**p[1] * self.lw_x**p[0], '-', label='$\zeta_{lw}$ = %0.2f $\pm$ %0.2f' % (p[0]/2, self.zeta_lw_err/2))
            self.ax_lw.plot([],[], label="$A_{lw} = %0.2f$" % (self.A_lw))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$w^2(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
    #
    def get_kolton_width(self, rmin_px=5, dr_px=1, overlap = 0.):
        """
        Doc Kolton's Width
        """
        N = len(self.u)
        possible_window_sizes = np.arange(rmin_px, N, dr_px)
        kwidth = []
        for window_size in possible_window_sizes:
            parts = 0
            kwidth_per_window = 0
            window_step = int(np.ceil(window_size*(1-overlap)))
            for i in xrange(0, N - window_size, window_step):
                m, b, _, _, _ = lr(self.z[i:i+window_size], self.u[i:i+window_size])
                kwidth_per_window += self.window_averaging(self.u[i:i+window_size] - self.z[i:i+window_size]*m - b)
                parts += 1
            kwidth.append(kwidth_per_window / parts)
        self.rkw = possible_window_sizes*self.umperpix
        self.kolton_width = np.array(kwidth)
        self.get_zeta_kw()
    #
    def get_zeta_kw(self, get_exponent=True, limits = []):
        """
        draws graph and accepts clicks to find range of slope.
        get_exponent=True brings calculated exponent.
        limits=[a,b] calculates exponent from those limits, given as integers
        corresponding to elements in x-array.
        """
        self.fig_kw, self.ax_kw = plt.subplots(1,1)
        self.ax_kw.plot(self.rkw[1:], self.kolton_width[1:], 'o-')
        plt.xscale('log')
        plt.yscale('log')
        self.total_clicks = 0
        self._kwfit = []
        if get_exponent and limits==[]:
            self.cidslope = self.fig_kw.canvas.mpl_connect('button_press_event', self.onclick_kw)
        elif not get_exponent and limits==[]:
            self.ax_kw.plot(self.kw_x, 10**(self.A_kw) * self.kw_x**(2*self.zeta_kw), '-', label='$\zeta_{kw}$ = %0.2f $\pm$ %0.2f' % (self.zeta_kw, self.zeta_kw_err))
            self.ax_kw.plot([],[], label="$A_{kw} = %0.2f$" % (self.A_kw))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$KW(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
        elif limits != []:
            ini, fin = limits
            self._kwfit = limits
            p, cov = np.polyfit( np.log10(self.rkw[ini:fin]), np.log10(self.kolton_width[ini:fin]), 1, cov=True)
            self.kw_x = np.logspace(np.log10(self.rkw[ini]), np.log10(self.rkw[-1]), 11)
            self.zeta_kw = p[0]/2
            self.zeta_kw_err = np.sqrt(cov[0,0])/2
            self.A_kw = 10**p[1]
            self.ax_kw.plot(self.kw_x, 10**p[1] * self.kw_x**p[0], '-', label='$\zeta_{kw}$ = %0.2f $\pm$ %0.2f' % (p[0]/2, self.zeta_kw_err/2))
            self.ax_kw.plot([],[], label="$A_{KW(r)} = %0.2f$" % (self.A_kw))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$KW(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
    #
    def onclick_kw(self, event):
        if event.inaxes is not None:
            self._kwfit.append(event.xdata)
            self.total_clicks += 1
        if self.total_clicks == 2:
            self.fig_kw.canvas.mpl_disconnect(self.cidslope)
            ini = np.abs(self.rkw - self._kwfit[0]).argmin()
            fin = np.abs(self.rkw - self._kwfit[1]).argmin()
            self._kwfit[0], self._kwfit[1] = ini, fin
            print 'ini-fin = %i-%i' % (ini,fin)
            p, cov = np.polyfit( np.log10(self.rkw[ini:fin]), np.log10(self.kolton_width[ini:fin]), 1, cov=True)
            self.kw_x = np.logspace(np.log10(self.rkw[ini]), np.log10(self.rkw[-1]), 11)
            self.zeta_kw = p[0]/2
            self.zeta_kw_err = np.sqrt(cov[0,0])/2
            self.A_kw = 10**p[1]
            self.ax_kw.plot(self.kw_x, 10**p[1] * self.kw_x**p[0], '-', label='$\zeta_{kw}$ = %0.2f $\pm$ %0.2f' % (p[0]/2, self.zeta_kw_err/2))
            self.ax_kw.plot([],[], label="$A_{kw} = %0.2f$" % (self.A_kw))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$KW(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
    #
    def get_B_r(self):
        #possible_window_sizes is the 'r' variable to graph
        N = len(self.u)
        possible_window_sizes = np.arange(1, N)
        B_r = []
        for windowSize in possible_window_sizes:
            Br_per_window = 0
            for i in xrange(N-windowSize):
                Br_per_window += (self.u[i+windowSize] - self.u[i]) ** 2
            B_r.append( Br_per_window / (N-windowSize) )
        self.B_r = np.array(B_r)
        self.rB_r = possible_window_sizes*self.umperpix
        self.get_zeta_B_r()
    #
    def get_zeta_B_r(self, get_exponent=True, limits = []):
        """
        draws graph and accepts clicks to find range of slope
        """
        self.fig_Br, self.ax_Br = plt.subplots(1,1)
        self.ax_Br.plot(self.rB_r, self.B_r, 'o-')
        plt.xscale('log')
        plt.yscale('log')
        self.total_clicks = 0
        self._brfit = []
        if get_exponent and limits == []:
            self.cidslope = self.fig_Br.canvas.mpl_connect('button_press_event', self.onclick_Br)
        elif not get_exponent and limits == []:
            self.ax_Br.plot(self.Br_x, 10**(self.A_B_r) * self.Br_x**(2*self.zeta_B_r), '-', label='$\zeta_{B(r)}$ = %0.2f $\pm$ %0.2f' % (self.zeta_B_r, self.zeta_B_r_err))
            self.ax_Br.plot([],[], label="$A_{B(r)} = %0.2f$" % (self.A_B_r))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$B(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
        elif limits != []:
            ini, fin = limits
            self._brfit = limits
            p, cov = np.polyfit( np.log10(self.rB_r[ini:fin]), np.log10(self.B_r[ini:fin]), 1, cov=True)
            self.Br_x = np.logspace(np.log10(self.rB_r[ini]), np.log10(self.rB_r[-1]), 11)
            self.zeta_B_r = p[0]/2
            self.zeta_B_r_err = np.sqrt(cov[0,0])/2
            self.A_B_r = 10**p[1]
            self.ax_Br.plot(self.Br_x, 10**p[1] * self.Br_x**p[0], '-', label='$\zeta_{B(r)}$ = %0.2f $\pm$ %0.2f' % (p[0]/2, self.zeta_B_r_err/2))
            self.ax_Br.plot([],[], label="$A_{B(r)} = %0.2f$" % (self.A_B_r))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$B(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
    #
    def onclick_Br(self, event):
        if event.inaxes is not None:
            self._brfit.append(event.xdata)
            self.total_clicks += 1
        if self.total_clicks == 2:
            self.fig_Br.canvas.mpl_disconnect(self.cidslope)
            ini = np.abs(self.rB_r - self._brfit[0]).argmin()
            fin = np.abs(self.rB_r - self._brfit[1]).argmin()
            self._brfit[0], self._brfit[1] = ini, fin
            print 'ini-fin = %i-%i' % (ini,fin)
            p, cov = np.polyfit( np.log10(self.rB_r[ini:fin]), np.log10(self.B_r[ini:fin]), 1, cov=True)
            self.Br_x = np.logspace(np.log10(self.rB_r[ini]), np.log10(self.rB_r[-1]), 11)
            self.zeta_B_r = p[0]/2
            self.zeta_B_r_err = np.sqrt(cov[0,0])/2
            self.A_B_r = 10**p[1]
            self.ax_Br.plot(self.Br_x, 10**p[1] * self.Br_x**p[0], '-', label='$\zeta_{B(r)}$ = %0.2f $\pm$ %0.2f' % (p[0]/2, self.zeta_B_r_err/2))
            self.ax_Br.plot([],[], label="$A_{B(r)} = %0.2f$" % (self.A_B_r))
            plt.xlabel('$r \, [\mu m]$')
            plt.ylabel('$B(r)\,[\mu m^2]$')
            plt.draw()
            plt.legend()
    #
    def get_structure_factor(self):
        print 'L/2\n'
        L = len(self.u) #u has units of um
        u_hat_pos = np.fft.fft(self.u)[:L/2]
        s = u_hat_pos*u_hat_pos.conj()
        self.S_q = s.real
        self.q = 2*np.pi/((L-1)*self.umperpix)*np.arange(L/2)
        self.get_zeta_S_q()
    #
    def get_zeta_S_q(self, get_exponent=True, limits = []):
        """
        draws graph and accepts clicks to find range of slope
        """
        self.fig_Sq, self.ax_Sq = plt.subplots(1,1)
        self.ax_Sq.plot(self.q, self.S_q, 'o-')
        plt.xscale('log')
        plt.yscale('log')
        self.total_clicks = 0
        self._sqfit = []
        if get_exponent and limits == []:
            self.cidslope = self.fig_Sq.canvas.mpl_connect('button_press_event', self.onclick_Sq)
        elif not get_exponent and limits == []:
            self.ax_Sq.plot(self.Sq_x, 10**(self.A_S_q) * self.Sq_x**(2*self.zeta_S_q), '-', label='$\zeta_{S(q)}$ = %0.2f' % (self.zeta_S_q))
            #label='$\zeta_{S(q)}$ = %0.2f $\pm$ %0.2f' % (self.zeta_S_q, self.zeta_S_q_err))
            plt.xlabel('$q [\mu m ^{-1}]$')
            plt.ylabel('$S(r) [\mu m ^{-2}]$')
            plt.draw()
            plt.legend()
        elif limits != []:
            ini, fin = limits
            self._sqfit = limits
            p, cov = np.polyfit( np.log10(self.q[ini:fin]), np.log10(self.S_q[ini:fin]), 1, cov=True)
            self.Sq_x = np.logspace(np.log10(self.q[ini]), np.log10(self.q[-1]), 11)
            self.zeta_S_q = (-p[0]-1)/2.
            self.zeta_S_q_err = (-np.sqrt(cov[0,0])-1)/2
            self.A_S_q = 10**p[1]
            self.ax_Sq.plot(self.Sq_x, 10**p[1] * self.Sq_x**p[0], '-', label='$\zeta_{S(q)}$ = %0.2f' % (self.zeta_S_q))
            #label='$\zeta_{S(q)}$ = %0.2f $\pm$ %0.2f' % (self.zeta_S_q, self.zeta_S_q_err))
            plt.xlabel('$q [\mu m ^{-1}]$')
            plt.ylabel('$S(r) [\mu m ^{-2}]$')
            plt.draw()
            plt.legend()
    #
    def onclick_Sq(self, event):
        if event.inaxes is not None:
            self._sqfit.append(event.xdata)
            self.total_clicks += 1
        if self.total_clicks == 2:
            self.fig_Sq.canvas.mpl_disconnect(self.cidslope)
            ini = np.abs(self.q - self._sqfit[0]).argmin()
            fin = np.abs(self.q - self._sqfit[1]).argmin()
            self._sqfit[0], self._sqfit[1] = ini, fin
            print 'ini-fin = %i-%i' % (ini,fin)
            p, cov = np.polyfit( np.log10(self.q[ini:fin]), np.log10(self.S_q[ini:fin]), 1, cov=True)
            self.Sq_x = np.logspace(np.log10(self.q[ini]), np.log10(self.q[-1]), 11)
            self.zeta_S_q = (-p[0]-1)/2.
            self.zeta_S_q_err = (-np.sqrt(cov[0,0])-1)/2
            self.A_S_q = p[1]
            self.ax_Sq.plot(self.Sq_x, 10**p[1] * self.Sq_x**p[0], '-', label='$\zeta_{S(q)}$ = %0.2f' % (self.zeta_S_q))
            #label='$\zeta_{S(q)}$ = %0.2f $\pm$ %0.2f' % (self.zeta_S_q, self.zeta_S_q_err))
            plt.xlabel('$q [\mu m ^{-1}]$')
            plt.ylabel('$S(r) [\mu m ^{-2}]$')
            plt.draw()
            plt.legend()
    #
    def save_data(self, dataname = '', figext = '.png'):
        """
        dataname, extensions and attributes will be set automatically.
        Example: "sample01BzBx".
        """
        if dataname == '':
            if '.' in self.filename:
                ini = self.filename.find('.')
                dataname = self.filename[:ini]
            else:
                dataname = self.filename
        #
        if self.global_width or self.zeta_lw or self.zeta_B_r or self.zeta_S_q:
            expfile = open(self.save_folder+'/'+dataname+"_exps.dat", 'w')
            titles = ''
            values = ''
            if self.global_width:
                titles += 'Global Width\t'
                values += '%f\t' % self.global_width
            if self.zeta_lw:
                titles += 'Zeta_LW\tA_LW\t'
                values += '%f\t%f\t' % (self.zeta_lw, self.A_lw)
            if self.zeta_kw:
                titles += 'Zeta_KW\tA_KW\t'
                values += '%f\t%f\t' % (self.zeta_kw, self.A_kw)
            if self.zeta_B_r:
                titles += 'Zeta_Br\tA_Br\t'
                values += '%f\t%f\t' % (self.zeta_B_r, self.A_B_r)
            if self.zeta_S_q:
                titles += 'Zeta_Sq\tA_Sq\t'
                values += '%f\t%f\t' % (self.zeta_S_q, self.A_S_q)
            titles += 'Angle\tCrop Box\n'
            values += '%f\t%s' % (self.angle, self.box)
            expfile.write(titles)
            expfile.write(values)
            expfile.close()
        #
        if not np.array_equal(self.z, np.array([])):
            datafile = open(self.save_folder+'/'+dataname+"_u.dat", "w")
            for i, x in enumerate(self.z):
                datafile.write(str(x) + '\t' + str(self.u[i]) + '\n')
            datafile.close()
        #
        if not np.array_equal(self.rlw, np.array([])):
            datafile = open(self.save_folder+'/'+dataname+"_lw.dat", 'w')
            datafile.write('%i\t%i\n' % (self._lwfit[0], self._lwfit[1]))
            for i, x in enumerate(self.rlw):
                datafile.write(str(x) + '\t' + str(self.local_width[i]) + '\n')
            datafile.close()
        #
        if not np.array_equal(self.rkw, np.array([])):
            datafile = open(self.save_folder+'/'+dataname+"_kw.dat", 'w')
            datafile.write('%i\t%i\n' % (self._kwfit[0], self._kwfit[1]))
            for i, x in enumerate(self.rkw):
                datafile.write(str(x) + '\t' + str(self.kolton_width[i]) + '\n')
            datafile.close()
        #
        if not np.array_equal(self.rB_r, np.array([])):
            datafile = open(self.save_folder+'/'+dataname+"_Br.dat", 'w')
            datafile.write('%i\t%i\n' % (self._brfit[0], self._brfit[1]))
            for i, x in enumerate(self.rB_r):
                datafile.write(str(x) + '\t' + str(self.B_r[i]) + '\n')
            datafile.close()
        #
        if not np.array_equal(self.q, np.array([])):
            datafile = open(self.save_folder+'/'+dataname+"_Sq.dat", 'w')
            datafile.write('%i\t%i\n' % (self._sqfit[0], self._sqfit[1]))
            for i, x in enumerate(self.q):
                datafile.write(str(x) + '\t' + str(self.S_q[i]) + '\n')
            datafile.close()
        #
        try:
            self.fig_original.savefig(self.save_folder+'/'+dataname+'_wall'+figext)
        except:
            pass
        try:
            self.fig_lw.savefig(self.save_folder+'/'+dataname+'_lw'+figext)
        except:
            pass
        try:
            self.fig_kw.savefig(self.save_folder+'/'+dataname+'_kw'+figext)
        except:
            pass
        try:
            self.fig_Br.savefig(self.save_folder+'/'+dataname+'_Br'+figext)
        except:
            pass
        try:
            self.fig_Sq.savefig(self.save_folder+'/'+dataname+'_Sq'+figext)
        except:
            pass
    #
#########################