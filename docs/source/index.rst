massphoto: from movies to masses
=================================================

**massphoto** is an open-source Python package for analysing mass photometry
movies, from detecting single-protein events to measuring contrasts and
calibrated mass distributions.

* **For mass photometry users:** Analyse your measurements directly in Python, keeping your entire workflow transparent, shareable, and fully under your control.
* **Simple yet flexible:** Turn raw movies into calibrated mass distributions with a single command, or customise parameters like integration time and PSF shape.
* **Open and reproducible:** Automate batch processing across large datasets and output single-molecule events as standard pandas DataFrames to integrate into your analysis pipelines.

.. container:: mp-home-actions

   .. button-ref:: usage
      :color: primary

      Get started

   .. button-ref:: installation
      :color: primary
      :outline:

      Install massphoto

   .. code-block:: bash
      :class: mp-home-pip-snippet

      pip install massphoto

..
   The four workflow panels below (raw.png, events.png, profile.svg, histogram.svg)
   are generated build outputs produced by _save_homepage_panels() in
   docs/source/_examples/02-mass_distribution.py from the public 006_dynamin_6.mp movie.
   Provenance metadata is stored in auto_examples/images/homepage/provenance.json.
   To update or regenerate them, edit the hidden block in example 02 and run an
   executing Sphinx build (without -D plot_gallery=0).

.. grid:: 2 2 4 4
   :gutter: 4
   :class-container: mp-home-workflow

   .. grid-item::
      :class: mp-home-step

      .. image:: auto_examples/images/homepage/raw.png
         :alt: Three grayscale frames from the same region of the measured movie.

      .. rubric:: 1. Raw movie

      Camera intensity over time

   .. grid-item::
      :class: mp-home-step

      .. image:: auto_examples/images/homepage/events.png
         :alt: The same measured region after ratiometric processing, with the fitted event marked.

      .. rubric:: 2. Reveal events

      Ratiometric background subtraction & detection

   .. grid-item::
      :class: mp-home-step

      .. image:: auto_examples/images/homepage/profile.svg
         :alt: Measured pixel profile through the marked event and its fitted PSF, including side lobes.

      .. rubric:: 3. Fit events

      Event positions and contrasts

   .. grid-item::
      :class: mp-home-step

      .. image:: auto_examples/images/homepage/histogram.svg
         :alt: Calibrated mass histogram of the full measurement with a fitted distribution.
         :target: auto_examples/02-mass_distribution.html

      .. rubric:: 4. Mass distribution

      Calibrate, filter, and analyse

Data from a real mass photometry measurement of dynamin (see the :ref:`detailed analysis tutorial <sphx_glr_auto_examples_02-mass_distribution.py>`).
The package reads raw mass photometry movies, performs ratiometric background subtraction, detects single-molecule events, fits their point spread functions (PSFs), and returns a table of fitted events with contrasts and calibrated masses.

.. admonition:: Best Practice Protocol & Data

   For comprehensive experimental guidelines, sample preparation protocols, and calibration strategies, see:
   Kratochvíl et al., *Best practice mass photometry: a guide to optimal single-molecule mass measurement*, Nature Protocols **2025**,
   `(10.1038/s41596-025-01255-4) <https://doi.org/10.1038/s41596-025-01255-4>`_.
   The benchmark datasets used in this documentation are part of this study and are available on `Zenodo (10.5281/zenodo.15800729) <https://doi.org/10.5281/zenodo.15800729>`_.

**massphoto** is designed for simplicity: a single command turns a movie into
a calibrated event table, while every stage - drift
correction, background subtraction, candidate detection, and PSF fitting - can be configured and customised.
Results are returned as standard pandas DataFrames with complete metadata:

.. code-block:: python

   import massphoto as mp
   df = mp.fit("./protein_measurement.mp")

Each row of the returned DataFrame describes one fitted single-molecule event:

.. raw:: html

   <div class="mp-dataframe-preview" role="region" aria-label="Sample event results" tabindex="0">
   <table class="dataframe">
     <thead>
       <tr style="text-align: right;">
         <th></th>
         <th>frame</th>
         <th>x</th>
         <th>y</th>
         <th>contrasts</th>
         <th>masses_kDa</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <th>0</th>
         <td>20</td>
         <td>86.09</td>
         <td>6.87</td>
         <td>0.00343</td>
         <td>101.2</td>
       </tr>
       <tr>
         <th>1</th>
         <td>20</td>
         <td>165.06</td>
         <td>72.34</td>
         <td>0.00604</td>
         <td>182.1</td>
       </tr>
       <tr>
         <th>2</th>
         <td>20</td>
         <td>205.38</td>
         <td>74.17</td>
         <td>0.00648</td>
         <td>195.8</td>
       </tr>
       <tr>
         <th>3</th>
         <td>20</td>
         <td>124.95</td>
         <td>80.87</td>
         <td>0.00580</td>
         <td>174.5</td>
       </tr>
       <tr>
         <th>4</th>
         <td>20</td>
         <td>189.78</td>
         <td>93.18</td>
         <td>0.01202</td>
         <td>367.4</td>
       </tr>
     </tbody>
   </table>
   </div>



To include mass columns, first set your
:ref:`mass calibration <sphx_glr_auto_examples_01-mass_calibration.py>` in
``mp.defaults``. Without calibration, you can still inspect fitted contrasts.

The :doc:`usage guide <usage>` explains the processing stages, result columns,
and choices that affect your analysis. The examples provide complete recipes.

Explore the code examples
-------------------------

Walk through runnable Python scripts and downloadable Jupyter notebooks covering
typical mass photometry workflows (see the full :doc:`example gallery <auto_examples/index>`):

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Calibrate a measurement
      :link: sphx_glr_auto_examples_01-mass_calibration.py
      :link-type: ref

      .. raw:: html

         <svg class="mp-home-icon" viewBox="0 0 32 32" aria-hidden="true" focusable="false"><path d="M5 4v23h23M8 23 25 7"/><circle cx="10" cy="21" r="2"/><circle cx="17" cy="15" r="2"/><circle cx="24" cy="8" r="2"/></svg>

      Code example showing how to turn known reference masses into a
      contrast-to-mass calibration and check the resulting
      mass scale.

   .. grid-item-card:: Analyse a sample
      :link: sphx_glr_auto_examples_02-mass_distribution.py
      :link-type: ref

      .. raw:: html

         <svg class="mp-home-icon" viewBox="0 0 32 32" aria-hidden="true" focusable="false"><path d="M4 5v23h25M9 25V15h4v10M17 25V7h4v18M25 25v-6h4v6"/></svg>

      Code example going from a raw movie to a calibrated mass distribution,
      including fit quality filtering, and peak fitting.

   .. grid-item-card:: Step-by-step pipeline
      :link: sphx_glr_auto_examples_07-step_by_step_pipeline.py
      :link-type: ref

      .. raw:: html

         <svg class="mp-home-icon" viewBox="0 0 32 32" aria-hidden="true" focusable="false"><path d="M7 4v8m0 6v10M16 4v15m0 6v3M25 4v3m0 6v15"/><rect x="4" y="12" width="6" height="6" rx="1"/><rect x="13" y="19" width="6" height="6" rx="1"/><rect x="22" y="7" width="6" height="6" rx="1"/></svg>

      Walk-through of each processing stage individually to inspect
      intermediate results and tune parameters for your measurement.

Feedback & Community
--------------------

Encountered a bug, working with an unsupported instrument or file format, or have a feature request?
Please open an issue on the `GitHub Issue Tracker <https://github.com/thielec/massphoto/issues>`_!

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   auto_examples/index
   api/index
   contributing

.. container:: mp-home-index-links

   :ref:`genindex` · :ref:`modindex` · :ref:`search`
