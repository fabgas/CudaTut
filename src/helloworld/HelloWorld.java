package helloworld;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class HelloWorld {
	  /**
     * The CUDA context created by this sample
     */
    private static CUcontext context;
    /**
     * The module which is loaded in form of a PTX file
     */
    private static CUmodule module;
    
    /**
     * The actual kernel function from the module
     */
    private static CUfunction function;
    
    /**
     * Temporary memory for the device output
     */
    private static CUdeviceptr deviceBuffer;
    
	public static void main(String[] args) {
		 // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        init();
        //initialisation d'un pointeur pour la mémoire d'inout
        float hostInput[] = createRandomArray(100);
        CUdeviceptr deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, hostInput.length * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), 
            hostInput.length * Sizeof.FLOAT);
        
        float resultJCuda = helloworld(deviceInput, hostInput.length);
    
        cuMemFree(deviceInput);
	}
	
	public static float helloworld( Pointer deviceInput, int length) {
		  Pointer kernelParameters = Pointer.to(
		            Pointer.to(deviceInput),
		            Pointer.to(new int[]{length})
		        );
		 
		  	int blocks = 10;
		  	int threads = 20; // threads par block ?
		    int sharedMemSize = threads * Sizeof.FLOAT;
		        // Call the kernel function.
		        cuLaunchKernel(function,
		            blocks,  1, 1,         // Grid dimension
		            threads, 1, 1,         // Block dimension
		            sharedMemSize, null,   // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
		        cuCtxSynchronize();
		        
		        return 0.0f;
	}
	/**
     * Initialize the driver API and create a context for the first
     * device, and then call {@link #prepare()}
     */
    private static void init()
    {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        prepare();
    }
    
    /**
     * Prepare everything for calling the reduction kernel function.
     * This method assumes that a context already has been created
     * and is current!
     */
    public static void prepare()
    {
        // Prepare the ptx file.
        String ptxFileName = null;
        try
        {
            ptxFileName = preparePtxFile("hello.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        // Load the module from the PTX file
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "reduce" function.
        function = new CUfunction();
        cuModuleGetFunction(function, module, "hello");
        
        // Allocate a chunk of temporary memory (must be at least
        // numberOfBlocks * Sizeof.FLOAT)
        deviceBuffer = new CUdeviceptr();
        cuMemAlloc(deviceBuffer, 1024 * Sizeof.FLOAT);
        
    }

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }
   
    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
    /**
     * Create an array of the given size, with random data
     * 
     * @param size The array size
     * @return The array
     */
    private static float[] createRandomArray(int size)
    {
        Random random = new Random(0);
        float array[] = new float[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = random.nextFloat();
        }
        return array;
    }
}
