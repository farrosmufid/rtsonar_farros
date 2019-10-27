# Import functions and libraries
import numpy as np
import matplotlib.cm as cm
from scipy import signal
from scipy import interpolate
from numpy import *
import threading,time, queue, pyaudio
import bokeh.plotting as bk
from bokeh.resources import INLINE
from bokeh.models import GlyphRenderer
from bokeh.io import push_notebook
from IPython.display import clear_output
import sys
bk.output_notebook(INLINE)


def put_data( Qout, ptrain, Twait, stop_flag):
    while( not stop_flag.is_set() ):
        if ( Qout.qsize() < 2 ):
            Qout.put( ptrain )
            
        time.sleep(Twait)
            
    Qout.put("EOT")
            
def play_audio( Qout, p, fs, stop_flag, dev=None):
    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    while ( not stop_flag.is_set()):
        data = Qout.get()
        if data is "EOT" :
            break
        try:
            ostream.write( data.astype(np.float32).tostring() )
        except:
            break
    ostream.stop_stream();
    ostream.close()
            
def record_audio( Qin, p, fs, stop_flag, dev=None,chunk=2048):
    istream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),input=True,input_device_index=dev,frames_per_buffer=chunk)

    # record audio in chunks and append to frames
    ct = 0
    frames = [];
    while (  not stop_flag.is_set() ):
        try:  # when the pyaudio object is destroyed, stops
            data_str = istream.read(chunk,exception_on_overflow=False) # read a chunk of data
            ct += 1
        except:
            print("Count is ",ct)
            print("Unexpected error:", sys.exc_info()[0])
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        Qin.put( data_flt ) # append to list
    istream.stop_stream();
    istream.close()
    Qin.put("EOT")

    
def signal_process( Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag  ):
    # Signal processing function for real-time sonar
    # Takes in streaming data from Qin and process them using the functions defined above
    # Uses the first 2 pulses to calculate for delay
    # Then for each Nseg segments calculate the cross correlation (uses overlap-and-add)
    # Inputs:
    # Qin - input queue with chunks of audio data
    # Qdata - output queue with processed data
    # pulse_a - analytic function of pulse
    # Nseg - length between pulses
    # Nplot - number of samples for plotting
    # fs - sampling frequency
    # maxdist - maximum distance
    # temperature - room temperature

    crossCorr = functions[2]
    findDelay = functions[3]
    dist2time = functions[4]
    
    # initialize Xrcv 
    Xrcv = zeros( 3 * Nseg, dtype='complex' );
    cur_idx = 0; # keeps track of current index
    found_delay = False;
    maxsamp = min(int(dist2time( maxdist, temperature) * fs), Nseg); # maximum samples corresponding to maximum distance
    
    while(  not stop_flag.is_set() ):
        
        # Get streaming chunk
        chunk = Qin.get();
        if (chunk is "EOT"):
            break;
        Xchunk =  crossCorr( chunk, pulse_a ) 
        Xchunk = np.reshape(Xchunk,(1,len(Xchunk)))
        
        # Overlap-and-add
        # If chunk is empty, add zeros
        try:
            Xrcv[cur_idx:(cur_idx+len(chunk)+len(pulse_a)-1)] += Xchunk[0,:];
        except:
            1
            #print("empty audio stream. Skipping.")
            #Xrcv[cur_idx:(cur_idx+len(chunk)+len(pulse_a)-1)] += Xchunk[0,:];
            
        cur_idx += len(chunk)
        if( found_delay and (cur_idx >= Nseg) ):
            # If delay has been found once (elif statement below) keep finding
            # This fixes drift on raspberry pi, but slows things down
            if found_delay:
                idx = findDelay( abs(Xrcv), Nseg );
                Xrcv = np.roll(Xrcv, -idx );
                Xrcv[-idx:] = 0;
                cur_idx = cur_idx - idx;
            
            # crop a segment from Xrcv and interpolate to Nplot
            # Divide by peak value (index 0), or a non-zero number in case audio buffer is empty
            Xrcv_seg = (abs(Xrcv[:maxsamp].copy()) / np.maximum(abs( Xrcv[0] ),1e-5) ) ** 0.5 ;   
            interp = interpolate.interp1d(r_[:maxsamp], Xrcv_seg)
            Xrcv_seg = interp( r_[:maxsamp-1:(Nplot*1j)] )
            
            # remove segment from Xrcv
            Xrcv = np.roll(Xrcv, -Nseg );
            Xrcv[-Nseg:] = 0
            cur_idx = cur_idx - Nseg;
            
            Qdata.put( Xrcv_seg );
            #Qdata.put(np.abs(Xchunk));
            
        elif( cur_idx > 2 * Nseg ):
            # Uses two pulses to calculate delay
            idx = findDelay( abs(Xrcv), Nseg );
            Xrcv = np.roll(Xrcv, -idx );
            Xrcv[-idx:] = 0;
            cur_idx = cur_idx - idx - 1;
            found_delay = True
             
    Qdata.put("EOT")
            
            
def image_update( Qdata, fig, Nrep, Nplot, stop_flag):
    renderer = fig.select(dict(name='echos', type=GlyphRenderer))
    source = renderer[0].data_source
    img = source.data['image'][0];
    
    while(  not stop_flag.is_set() ):
        new_line = Qdata.get();
        #new_line = np.uint8(new_line/np.max(new_line)*255)
        
        if new_line is "EOT" :
            break
       
        #print(np.percentile(new_line,99))
        #print(np.max(new_line))
        # Normalize to some percentile instead of max, prevent divide by 0 and apply gamma to help see dim peaks
        new_line = np.minimum(new_line/np.maximum(np.percentile(new_line,97),1e-5),1)**(1/1.8)
        img = np.roll( img, 1, 0);
        view = img.view(dtype=np.uint8).reshape((Nrep, Nplot, 4))
        view[0,:,:] = cm.jet(new_line)*255;
    
        source.data['image'] = [img]
        push_notebook()
        Qdata.queue.clear();
        
    

        
def rtsonar( f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature, functions ):

    clear_output();
    genChirpPulse = functions[0]
    genPulseTrain = functions[1]
    
    pulse_a = genChirpPulse(Npulse, f0,f1,fs)
    hanWin = np.hanning(Npulse)
    # hanWin = np.reshape(hanWin, (Npulse,1) )
    pulse_a = np.multiply(pulse_a,hanWin)
    pulse = np.real(pulse_a)
    ptrain = genPulseTrain(pulse, Nrep, Nseg)
    
    # create an input output FIFO queues
    Qin = queue.Queue()
    Qout = queue.Queue()
    Qdata = queue.Queue()

    # create a pyaudio object
    p = pyaudio.PyAudio()
    
    # create black image
    img = np.zeros((Nrep,Nplot), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((Nrep, Nplot, 4))
    view[:,:,3] = 255;
    
    # initialize plot
    fig = bk.figure(title = 'Sonar',  y_axis_label = "Time [s]", x_axis_label = "Distance [cm]",
                    x_range=(0, maxdist), y_range=(0, Nrep * Nseg / fs ) , 
                    plot_height = 400, plot_width = 800 )
    fig.image_rgba( image = [ img ], x=[0], y=[0], dw=[maxdist], dh=[Nrep * Nseg / fs ], name = 'echos' )
    bk.show(fig,notebook_handle=True) # add notebook_handle=True to make it update

    # initialize stop_flag
    stop_flag = threading.Event()

    # initialize threads
    t_put_data = threading.Thread(target = put_data,   args = (Qout, ptrain, Nseg / fs*3, stop_flag  ))
    t_rec = threading.Thread(target = record_audio,   args = (Qin, p, fs, stop_flag  ))
    t_play_audio = threading.Thread(target = play_audio,   args = (Qout, p, fs, stop_flag  ))
    t_signal_process = threading.Thread(target = signal_process, args = ( Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag))
    t_image_update = threading.Thread(target = image_update, args = (Qdata, fig, Nrep, Nplot, stop_flag ) )

    # start threads
    t_put_data.start()
    t_rec.start()
    #record_audio(Qin, p, fs, stop_flag)
    t_play_audio.start()
    t_signal_process.start()
    t_image_update.start()

    return stop_flag
    

