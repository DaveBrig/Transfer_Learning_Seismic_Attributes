# Transfer Learning with Seismic Attributes

This post demonstrates that computer vision methods are a powerful means for extracting important information from geophysical imagery, but expertise is crucial when preparing the images for analysis. Since computer vision methods are highly sensitive to textural variation, subject matter experts should carefully consider how image preparation intends to enhance or alternatively dampen geological features that are of interest. 

For this demonstration, we applied three different treatments to an example geophysical image. Each of these images were subjected to the same transformations and feature extraction pipeline. The features were then passed through a dimension reduction and clustering process. The results show that preparation of an image has a profound impact on the quality of extracted features and the resulting geological information to be gleaned. 

## Image Preparation

We created an attribute extracted from a 3D Marine Seismic data. Seismic data is ideal for analysing textural variations in geology. Different rock types have different densities, and sound waves emitted during seismic acquisition travel through different rock types at different velocities. Together variations in rock density and seismic velocity cause changes in acoustic impedance which manifests as changes in reflectivity in seismic data. Such variation in reflectivity can help interpreters to visually identify boundaries between seismic facies which can often represent transitions between rock types.      

We picked the 'Top of Thylacine' horizon on the Investigator 3D Marine Seismic Survey (MSS), and used the interpretation to create a surface. The Thylacine member represents a regional interface between thick Cretaceous aged marine claystones and underlying shore-face sandstones. It is the top of a regional reservoir horizon that hosts significant gas resources in offshore Victoria. The Top of Thylacine TWT structure is shown in the figure below.  

![Location](images/Location_map.png)

The interface at the Top of Thylacine surface causes a strong impedance contrast and is clearly discernible in seismic imagery. To generate a textural representation of the shore-face sandstones known to be present beneath this surface, the Incoherence attribute was calculated within a search window of 50 millisenconds beneath the Top of Thylacine horizon. This attribute is ideal for visualising changes in texture caused by differing facies or the presence of faults. The Incoherence attribute is derived by quantifying the difference between neighbouring traces (DUG Insight User Manual, n.d.). Similar traces have values near to zero while high values represent traces that are dissimilar. The attribute can therefore highlight boundaries between different rock types and/or faults. In the cross-section shown below the consistently high amplitude reflections beneath the Top of the Thylacine surface (labelled 'Top Porosity') indicates a contiguous sand body with a low incoherence. However, to the left of the section, this character gives way to lower amplitude and inconsistent reflections which probably represent a change to marine mudstone on the paleo continental slope deposited out-board of the sand dominated marine shelf. The variability in the seismic character of this out-board section would produce a high Incoherence.  

![Section](images/Geograph.png)

For this experiment three different treatments were applied to the Incoherence attribute: 

1. Rendered as a single-band greyscale image
2. A 'Viridis' colour bar applied, colour scale truncated to a range of 0-0.1, and the resulting image rendered as a multi-band RGB
3. Finally, to enhance texture a multi-directional hillshade was applied to the coloured rgb image. This has the effect of enhancing the high Incoherence zones which could represent boundaries between rock types or faults.

![Section](images/Incoherence.png)

## Feature Extraction

To conduct a textural analysis of each of the alternative geophysical images, a pre-trained Convolutional Neural Network (CNN) known as ResNet50 was used to extract features from each variation. ResNet50 is an open-source deep CNN that can be used for image classification and other computer vision modelling tasks. It was trained using millions of images from the ImageNet database which provides image data for researchers to train object recognition models (ImageNet, 2021). The lower layers of ResNet50 capture general visual features like edges, textures, and basic shapes. These features can be used for tasks such as dimension reduction and unsupervised machine learning.

To perform feature extraction, each of the geophysical image variations were split into a series of 3265 individual 600x600m tiles. Each of the image tiles were then passed through the pre-trained ResNet50 architecture. Before allowing the architecture to proceed to the final layer and perform classification, the features were extracted from the penultimate layer. Each of the 2064 features are numerical representations of textural information for each of the image tiles. Assuming that the features are of good quality, these can be used to quantify differences between the image tiles based on texture. 

The feature extraction process has transformed each image tile into a multi-dimensional dataset. These multi-dimensional datasets will be used to compare the three image variations based on the the textural information extracted by the pre-trained network.     

## Dimension Reduction

To compare the feature sets between the geophysical images, a process of dimension reduction followed by clustering was performed. The objective was to determine firstly whether textural domains can be distinguished from each of the image variations, and then which of the variations provide the best representation of the sub-surface and are therefore most useful for distinguishing geological domains.  

For dimension reduction each of the feature sets were scaled using the python Standard Scalar, before applying Principal Component Analysis (PCA) and finally Uniform Manifold and Projection (UMAP) to reduce each of the feature sets to three components.   

As can be seen below, each of the UMAP representations are slightly different, indicating a difference in the ability for the extracted feature sets to distinguish between textural domains. To assist with comparisons, all tiles that fall within the Thylacine gas field are coloured in purple and those proximal to the dry well Geographe North-1 are highlighted in red. The objective of this experiment is not to predict hydrocarbon distribution, but since the gas is known to be reservoired within shallow marine sands of the Thylacine member, this visualisation can assist with identifying textural domains that are consistent with this facies. 

![Section](images/UMAP.png)

## Unsupervised Machine Learning

After dimension reduction, the Louvain clustering method was implemented in an attempt to distinguish between textural domains. Louvain clustering is a popular unsupervised clustering method used for community detection in networks or graphs (Lu et al. 2014). This involves identifying groups of data within a network that are more densely connected to each other than to data in other groups.

In the visualisation below, the clusters have been re-ordered based on the location of the cluster in UMAP space. Clusters associated with the gas bearing reservoir quality sands have the warmest (yellow) colour, while those most distant from the gas bearing sands in UMAP space have cooler colours (blue, purple). 

![Section](images/clusters.png)

The yellow coloured cluster from all of the feature sets seem to capture the reservoir sands at Thylacine and perhaps highlight other similar unclassified sands. However, it is difficult to fully assess the differences between feature sets until the clusters can be visualised and compared spatially. 

Below is a visualisation of each cluster set across the Investigator 3D survey area. The clusters associated with the Greyscale image seem to highlight the known gas fields at Thylacine, Geographe and Artisan indicating that the features derived from this image would probably perform well in a hydrocarbon classification model. However, the cluster sets associated with the coloured RGB and hillshaded images seem to exhibit more coherent geological bodies. The brightest colours of the clusters associated with the hillshaded image seem to conform well with low Incoherence zones, providing a plausible representation of the marine shelf sands known to exist across the survey area. Of the three feature sets, these clusters also show the clearest distinction between textural domains and probable geological features. 

These observations show that adjusting the appearance of a geophysical image can strongly affect the features extracted by a pre-trained CNN. Enhancing texture associated with high Incoherence using a multi directional hillshade has facilitated the feature extraction process to discriminate textural domains more clearly. This implies that the feature extraction process can be optimised if subject matter experts first take steps to enhance textures associated with desired geological targets.    

![Section](images/maps.png)

## Acknowledgments
Thankyou to 3D Energi Limited for providing the Investigator 3D MSS dataset and assisting with producing seismic attributes for the purpose of this experiment.  

## References
D.C.B. Cliff, S.C. Tye & Taylor R, (2004). The Thylacine and Geographe gas discoveries, offshore eastern Otway Basin. APPEA Journal. 44. 441-461. 10.1071/AJ03017. 

DUG Insight User Manual, n.d. https://help.dugeo.com/m/Insight/l/438797-incoherence 

ImageNet, (2021), https://www.image-net.org/update-mar-11-2021.php

Lu. H, Halappanavar. M & Kalyanaraman. A (2014),Parallel Heuristics for Scalable Community Detection. Parallel Computing: SI: Scientific graph analysis







