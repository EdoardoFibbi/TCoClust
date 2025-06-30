import numpy as np
from .plots import plot_coclust, _highlight_cell
from .utils import block_sort, part_matrix_to_dict


# class to represent result of procedure as object
# implements also plotting and summary methods
class TccResult:
    """ class 'TccResult' implements an object for the output of co-clustering procedures and implements methods for
        its construction, manipulation and visualisation.
        
        Attributes
        ----------
        BlockParameters : np.ndarray or tuple of np.ndarray (in the normal case)
            Estimated block parameters of the model.
        MixingProportions : dict
            Row and column mixing proportions, structured as:
                {Pi : np.ndarray,
                 Rho : np.ndarray
                 }
        Partitions : dict
            Row and column partition matrices, structured as:
                {Z : np.ndarray,
                 W : np.ndarray
                 }
        Cellwise : bool
            Whether the co-clustering method used cellwise trimming or not
        M : tuple
            Mask matrices, if the method employed cellwise trimming (self.Method = "cell_TRICC").
            In particular:
                * M[0] encodes missing cells (as M[0] == 0)
                * M[1] encodes flagged (outlying) cells (as M[1] == 0)
        logL : list
            Classification log-likelihood per iteration (of the best initialisation)
        Metric : list
            Metric values per iteration, when computed
            (SSE if density='Normal', Phi2 if density='Poisson')
        Success : bool
            True if algorithm produced non-degenerate solution
        ElapsedTime : float
            Total elapsed time, in seconds
        Input : TCoClust.TccInput
            Input parameters that produced the result
        
        
        Methods
        -------
        __init__ : constructor
            Initialises an object of this class
        plot : plotting function
            Plots the resulting co-clustering (see also plot_coclust)
        plotCellOut : plotting function for diagnostics of cellwise outliers
            Displays various cellwise outlier plots (e.g., robust vs non-robust cellwise residuals or posteriors)
        summary : display function
            Prints summary of co-clustering results
        
    """

    def __init__(self, **kwargs):

        self.BlockParameters = None
        self.MixingProportions = None
        self.Partitions = None
        self.M = None
        self.Cellwise = None
        self.logL = None
        self.Metric = None
        self.Success = None
        self.ElapsedTime = None
        self.Input = None

        if kwargs is not None:
            for key, value in kwargs.items():
                if key in self.__dict__:
                    setattr(self, key, value)
                else:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def plot(self,
             row_names=None,
             col_names=None,
             vmin=None,
             vmax=None,
             figsize=None,
             cmap="viridis",
             colorbar=False,
             transpose=False,
             keep_aspect=False,
             return_graphics_objects=False,
             fig=None,
             ax=None):

        """ plot is a method implementing the automatic plotting of the co-clustering result. The original data matrix
            is sorted according to the retrieved partitions and a colormap is used to encode cell values. Trimming is
            also automatically represented in the plot.

        Parameters
        ----------
        row_names : list or None
            Defaults to None
        col_names : list or None
            Defaults to None
        vmin : float or None
            Lowest value represented by the colormap used in plt.imshow
            If None, it is determined by the data range
            Defaults to None
        vmax : float or None
            Highest value represented by the colormap used in plt.imshow
            If None, it is determined by the data range
            Defaults to None
        figsize : tuple of two floats or None
            Size of figure, in inches
            If None, Matplotlib default is used
            Defaults to None
        cmap : string
            Must be a colormap supported by Matplotlib
            Defaults to "viridis"
        colorbar : bool
            If True, a colorbar is shown
            Defaults to False
        transpose : bool
            If True, the data matrix is transposed
            Defaults to False
        keep_aspect : bool
            If True, the data matrix will be represented using an equal scale on the rows and columns
            If False, the plot's ratio is adapted to the size of the figure
            Defaults to False
        return_graphics_objects : bool
            Whether to return the figure, axis (and, if present, colorbar) objects
            Defaults to False
        fig : plt.Figure or None
            To pass a Figure object to be used for plotting by the function
            Defaults to None
        ax : plt.Axes or None
            To pass an Axes object to be used for plotting by the function
            Defaults to None

        Returns
        -------
        None if return_graphics_objects is False; if True, returns a tuple of a figure, axis, and (if colorbar is True)
        a colorbar.

        """

        if self.Partitions is None:
            return

        # todo: add option to change colors of partition lines and highlighted cells
        graphics_objects = plot_coclust(self.Input.X, self.Partitions['Z'], self.Partitions['W'], row_names=row_names,
                                        col_names=col_names, vmin=vmin, vmax=vmax, figsize=figsize, cmap=cmap,
                                        colorbar=colorbar, transpose=transpose, keep_aspect=keep_aspect,
                                        return_graphics_objects=True, fig=fig, ax=ax)
        if self.Cellwise:
            n, p = self.Input.X.shape
            _, rowLabSort, colLabSort = block_sort(self.Input.X,
                                                   part_matrix_to_dict(self.Partitions["Z"]),
                                                   part_matrix_to_dict(self.Partitions["W"]),
                                                   list(range(n)),
                                                   list(range(p)))

            for i, j in zip(np.where(self.M[1] < .5)[0], np.where(self.M[1] < .5)[1]):
                ii = np.where(np.array(rowLabSort) == i)[0][0]
                jj = np.where(np.array(colLabSort) == j)[0][0]
                _highlight_cell(jj, ii, graphics_objects[1], color="red", linewidth=1., alpha=1)
            pass

        if return_graphics_objects:
            return graphics_objects

        return

    def summary(self, short=False):

        """ summary
        
        """

        print(f"\nTrimmed co-clustering procedure with {self.Input.density} block distribution executed.\n")
        print("Total elapsed time: {:.5g} s\n".format(self.ElapsedTime))
        print("Convergence to non-degenerate partition:", self.Success)

        if not short:

            print("\nResults summary:")
            print("________________")

            print("\nEstimated block parameters:\n")
            if self.Input.density == "Normal":
                print(
                    "\n".join("{}:\n{}".format(k, v) for k, v in zip(["\u03BC", "\u03C3\u00B2"], self.BlockParameters)))
            elif self.Input.density == "Poisson":
                print("\u03B1:\n{}".format(self.BlockParameters))

            print("\nEstimated mixing proportions:\n")
            print("\n".join(
                "{}:\t{}".format(k, v) for k, v in zip(["\u03C0", "\u03C1"], self.MixingProportions.values())))

            print("\nFinal value of model log-likelihood:", self.logL[-1])

            if (self.Input.density == "Normal") and (self.Metric is not None) and (len(self.Metric) != 0):
                print("\nFinal value of SSE:", self.Metric[-1])
            elif (self.Input.density == "Poisson") and (self.Metric is not None) and (len(self.Metric) != 0):
                print("\nFinal value of Phi2:", self.Metric[-1])

            print("\nInput summary:")
            print("______________")

            print("\nModel constraints:")
            if self.Input.density == "Normal":
                print("\n\tBlock constraints:", self.Input.constrained)
            print("\n\tMixing proportion constraints:", self.Input.equal_weights)

            print("\nPartition sizes: g = {} (rows), m = {} (columns)".format(self.Input.g, self.Input.m))

            print("\nTrimming levels: a = {:.3g} (rows), b = {:.3g} (columns)".format(self.Input.a, self.Input.b))

            print("\nInitialisation strategy:", self.Input.init_strategy)

        return


# class to represent input of procedure as object
# for now mostly for internal use, to be further developed
class TccInput:
    """ class 'TccInput' abstracts the input of the co-clustering procedure.
        
        Attributes
        ----------
        X : np.ndarray
            Data matrix
        g : int
            Number of row groups
        m : int
            Number of column groups
        density : str
            Data distribution (currently supported options: "Poisson" for counting data, "Normal" for 
            continuous data)
        constrained : bool
            If :code:`True`, in the Normal LBM block variances and mixing proportions are assumed to be equal.
            In the Poisson LBM the mixing proportions are assumed equal if :code:`True`.
            Defaults to :code:`False`.
        equal_weights : bool
            If :code:`True`. mixing proportions are assumed to be equal. 
            Overwrites `constrained` if :code:`density == "Poisson"`
            Defaults to :code:`False`.
        a : float
            Row trimming level, must be between 0 and 0.5
        b : float
            Column trimming level, must be between 0 and 0.5
        n_init : int (optional)
            Number of initializations of the algorithm.
            Defaults to 20.
        init_strategy : str or None (optional)
            Determines initialization strategy, which can be:
                - random if init_strategy==None or init_strategy=="rand"
                - trimmed double k-means if init_strategy=="dkm"
                - constrained trimmed Poisson LBM if init_strategy=="poic"
                - 'tandem' application of trimmed k-means if init_strategy=="clust" 
                  (requires Python module 'matlab', MATLAB engine and FSDA Toolbox)
            Defaults to None.
        init_params : str or np.ndarray or tuple (optional)
            Initialization of the block parameters:
                - if init_params=="sample", the block parameters are estimated from the data
                - if a g-by-m matrix is provided, it is used as initialization
                - else, a ValueError is raised
            Defaults to "sample".
        criterion : str (optional)
            Defines the stopping criterion of the algorithm, which can be based on
            the number of iterations ("iter"), on the likelihood ("tol"), or on both.
            Defaults to "both".
        t_max : int (optional)
            Maximum number of iterations of the algorithm.
            Defaults to 20.
        tol : float (optional)
            Stopping criterion based on the absolute difference of the likelihood function
            between two consecutive iterations of the algorithm.
            Defaults to 1.e-16.
        until_converged : bool (optional)
            If True and no non-degenerate solution is found for the n_init initializations, then
            the algorithm is restarted until a non-degenerate solution is found.
            Defaults to :code:`False`.
        simpleselect : bool (optional)
            If True a C implementation of simpleselect instead of NumPy's argpartition is used
            to find the [a*n]-th ([b*p]-th) ordered statistic for row (column) trimming. Requires simpleselect.so 
            in "C" folder on the working directory (see line 93).
            Defaults to :code:`False`.
        seed : None or int (optional)
            Seed of random number generator, defaults to :code:`None`. Set `seed` to a positive integer for
            reproducibility.
        
        
        
        Methods
        -------
        __init__ : constructor
            Initialises an object of this class
        
    """

    def __init__(self,
                 X,
                 g,
                 m,
                 density,
                 constrained=False,
                 equal_weights=False,
                 a=0,
                 b=0,
                 alpha=0,
                 beta=0,
                 n_init=0,
                 init_strategy=None,
                 init_params="sample",
                 criterion="both",
                 t_max=0,
                 t_burn_in=0,
                 tol=None,
                 until_converged=False,
                 simpleselect=False,
                 seed=None,
                 CleanedX=None):
        self.X = X
        self.CleanedX = CleanedX
        self.g = g
        self.m = m
        self.density = density
        self.constrained = constrained
        self.equal_weights = equal_weights
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.n_init = n_init
        self.init_strategy = init_strategy
        self.init_params = init_params
        self.criterion = criterion
        self.t_max = t_max
        self.t_burn_in = t_burn_in
        self.tol = tol
        self.until_converged = until_converged
        self.simpleselect = simpleselect
        self.seed = seed
        self.trim_quantile = False
        self.trim_global = False
