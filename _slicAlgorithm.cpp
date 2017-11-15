#include "_slicPrecomp.hpp"

using namespace std;

namespace cv
{
namespace slicNora
{

class SlicImpl : public SuperpixelSlic
{
public:
  SlicImpl(InputArray image, int algorithm, int region_size, float ruler);

  virtual ~SlicImpl();

  // perform amount of iteration
  virtual void iterate(int num_iterations = 10);

  // get amount of superpixels
  virtual int getNumberOfSuperpixels() const;

  // get image with labels
  virtual void getLabels(OutputArray labels_out) const;
  virtual void getUniforms(OutputArray uniforms) const;

  // get mask image with contour
  virtual void getLabelContourMask(OutputArray image, bool thick_line = true) const;

  // enforce connectivity over labels
  virtual void enforceLabelConnectivity(int min_element_size = 25);

protected:
  // image width
  int widthImg;

  // image width
  int heightImg;

  // image channels
  int channelsNo;

  // algorithm
  int m_algorithm;

  // region size
  int regionLength;

  // compactness
  float compactness;

  // ratio (MSlic)
  float m_ratio;

  // split (MSlic)
  float m_split;

  // current iter
  int m_cur_iter;

  // current iter
  int m_iterations;

private:
  // labels no
  int kSuperpixels;

  // stacked channels
  // of original image
  vector<Mat> channels;

  // seeds on x
  vector<float> seedsX;

  // seeds on y
  vector<float> seedsY;

  // labels storage
  Mat labels;

  // uniforms
  Mat uniforms;

  // seeds storage
  vector<vector<float> > seedsC;

  // adaptive k (MSlic)
  vector<float> m_adaptk;

  // merge threshold (MSlic)
  float m_merge;

  // initialization
  inline void initialize();

  // detect edges over all channels
  inline void DetectChEdges(Mat &edgemag);

  // random perturb seeds
  inline void PerturbSeeds(const Mat &edgemag);

  // fetch seeds
  inline void GetChSeedsS();

  // fetch seeds
  inline void GetChSeedsK();

  // Slic
  inline void PerformSlic(const int &num_iterations);

  // Slico
  inline void PerformSlico(const int &num_iterations);

  // MSlic
  inline void PerformMSlic(const int &num_iterations);

  // MSlic
  inline void SuperpixelSplit();
};

CV_EXPORTS Ptr<SuperpixelSlic> createSuperpixelSlic(InputArray image, int algorithm, int region_size, float ruler)
{
  return makePtr<SlicImpl>(image, algorithm, region_size, ruler);
}

SlicImpl::SlicImpl(InputArray _image, int _algorithm, int _region_size, float _ruler)
    : m_algorithm(_algorithm), regionLength(_region_size), compactness(_ruler)
{
  if (_image.isMat())
  {
    Mat image = _image.getMat();

    // image should be valid
    CV_Assert(!image.empty());

    // initialize sizes
    widthImg = image.size().width;
    heightImg = image.size().height;
    channelsNo = image.channels();

    // intialize channels
    split(image, channels);
  }
  else if (_image.isMatVector())
  {
    _image.getMatVector(channels);

    // array should be valid
    CV_Assert(!channels.empty());

    // initialize sizes
    widthImg = channels[0].size().width;
    heightImg = channels[0].size().height;
    channelsNo = (int)channels.size();
  }
  else
    CV_Error(Error::StsInternal, "Invalid InputArray.");

  // init
  initialize();
}

SlicImpl::~SlicImpl()
{
  channels.clear();
  seedsC.clear();
  seedsX.clear();
  seedsY.clear();
  labels.release();
}

int SlicImpl::getNumberOfSuperpixels() const
{
  return kSuperpixels;
}

void SlicImpl::initialize()
{
  // total amount of superpixels given its size as input
  kSuperpixels = int(float(widthImg * heightImg) / float(regionLength * regionLength));

  // initialize seed storage
  seedsC.resize(channelsNo);

  // intitialize label storage
  labels = Mat(heightImg, widthImg, CV_32S, Scalar::all(0));

  // storage for edge magnitudes
  Mat edgemag = Mat(heightImg, widthImg, CV_32F, Scalar::all(0));

  // perturb seeds is not absolutely necessary,
  // one can set this flag to false
  bool perturbseeds = true;

  if (perturbseeds)
    DetectChEdges(edgemag);

  if (m_algorithm == Slico)
    GetChSeedsK();
  else if ((m_algorithm == Slic) ||
           (m_algorithm == MSlic))
    GetChSeedsS();
  else
    CV_Error(Error::StsInternal, "No such algorithm");

  // update amount of labels now
  kSuperpixels = (int)seedsC[0].size();

  // perturb seeds given edges
  if (perturbseeds)
    PerturbSeeds(edgemag);

  if (m_algorithm == MSlic)
  {
    m_merge = 4.0f;
    m_adaptk.resize(kSuperpixels, 1.0f);
  }
}

void SlicImpl::iterate(int num_iterations)
{
  // store total iterations
  m_iterations = num_iterations;

  if (m_algorithm == Slico)
    PerformSlico(num_iterations);
  else if (m_algorithm == Slic)
    PerformSlic(num_iterations);
  else if (m_algorithm == MSlic)
    PerformMSlic(num_iterations);
  else
    CV_Error(Error::StsInternal, "No such algorithm");

  // re-update amount of labels
  kSuperpixels = (int)seedsC[0].size();
}

void SlicImpl::getLabels(OutputArray labels_out) const
{
  labels_out.assign(labels);
}

void SlicImpl::getUniforms(OutputArray uniformsOut) const
{
  cout << endl;
  Mat M(heightImg, widthImg, CV_8UC3, Scalar(0, 0, 0));

  for (int i = 0; i < heightImg; i++)
  {
    for (int j = 0; j < widthImg; j++)
    {
      int n = seedsC[0].at(labels.at<int>(i, j));
      int n1 = seedsC[1].at(labels.at<int>(i, j));
      int n2 = seedsC[2].at(labels.at<int>(i, j));

      M.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(n);
      M.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(n1);
      M.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(n2);
    }
  }

  // for (int i = 0; i < seedsX.size(); i++)
  // {
  //   cout << (int)i << "\t";
  // }
  // cout<<endl;

  // for (int i = 0; i < seedsX.size(); i++)
  // {
  //   uchar n = saturate_cast<uchar>(seedsX[i]);
  //   cout << (int)n << "\t";
  // }
  // cout << endl;

  // for (int i = 0; i < seedsY.size(); i++)
  // {
  //   uchar n = saturate_cast<uchar>(seedsY[i]);
  //   cout << (int)n << "\t";
  // }
  // cout << endl;

  // for (int k = 0; k < 3; k++)
  // {
  //   for (int i = 0; i < seedsC[0].size(); i++)
  //   {
  //     uchar n = saturate_cast<uchar>(seedsC[k][i]);
  //     cout << (int)n << "\t";
  //   }
  //   cout << endl;
  // }
  // cout<<endl;

  uniformsOut.assign(M);
}

void SlicImpl::getLabelContourMask(OutputArray _mask, bool _thick_line) const
{
  // default width
  int line_width = 2;

  if (!_thick_line)
    line_width = 1;

  _mask.create(heightImg, widthImg, CV_8UC1);
  Mat mask = _mask.getMat();

  mask.setTo(0);

  const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
  const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  int sz = widthImg * heightImg;

  vector<bool> istaken(sz, false);

  int mainindex = 0;
  for (int j = 0; j < heightImg; j++)
  {
    for (int k = 0; k < widthImg; k++)
    {
      int np = 0;
      for (int i = 0; i < 8; i++)
      {
        int x = k + dx8[i];
        int y = j + dy8[i];

        if ((x >= 0 && x < widthImg) && (y >= 0 && y < heightImg))
        {
          int index = y * widthImg + x;

          if (false == istaken[index])
          {
            if (labels.at<int>(j, k) != labels.at<int>(y, x))
              np++;
          }
        }
      }
      if (np > line_width)
      {
        mask.at<char>(j, k) = (uchar)255;
        istaken[mainindex] = true;
      }
      mainindex++;
    }
  }
}

/*
 * EnforceLabelConnectivity
 *
 *   1. finding an adjacent label for each new component at the start
 *   2. if a certain component is too small, assigning the previously found
 *      adjacent label to this component, and not incrementing the label.
 *
 */
void SlicImpl::enforceLabelConnectivity(int min_element_size)
{

  if (min_element_size == 0)
    return;
  CV_Assert(min_element_size >= 0 && min_element_size <= 100);

  vector<float> adaptk(kSuperpixels, 1.0f);

  if (m_algorithm == MSlic)
  {
    adaptk.clear();
  }

  const int dx4[4] = {-1, 0, 1, 0};
  const int dy4[4] = {0, -1, 0, 1};

  const int sz = widthImg * heightImg;
  const int supsz = sz / kSuperpixels;

  int div = int(100.0f / (float)min_element_size + 0.5f);
  int min_sp_sz = max(3, supsz / div);

  Mat nlabels(heightImg, widthImg, CV_32S, Scalar(INT_MAX));

  int label = 0;
  vector<int> xvec(sz);
  vector<int> yvec(sz);

  // MSlic
  int currentlabel;
  float diffch = 0.0f;
  vector<float> adjch;
  vector<float> curch;
  map<int, int> hashtable;

  if (m_algorithm == MSlic)
  {
    hashtable[-1] = 0;
    adjch.resize(channelsNo, 0);
    curch.resize(channelsNo, 0);
  }

  //adjacent label
  int adjlabel = 0;

  for (int j = 0; j < heightImg; j++)
  {
    for (int k = 0; k < widthImg; k++)
    {
      if (nlabels.at<int>(j, k) == INT_MAX)
      {
        nlabels.at<int>(j, k) = label;
        //--------------------
        // Start a new segment
        //--------------------
        xvec[0] = k;
        yvec[0] = j;
        currentlabel = labels.at<int>(j, k);
        //-------------------------------------------------------
        // Quickly find an adjacent label for use later if needed
        //-------------------------------------------------------
        for (int n = 0; n < 4; n++)
        {
          int x = xvec[0] + dx4[n];
          int y = yvec[0] + dy4[n];
          if ((x >= 0 && x < widthImg) && (y >= 0 && y < heightImg))
          {
            if (nlabels.at<int>(y, x) != INT_MAX)
            {
              adjlabel = nlabels.at<int>(y, x);
              if (m_algorithm == MSlic)
              {
                for (int b = 0; b < channelsNo; b++)
                {
                  adjch[b] = seedsC[b][labels.at<int>(y, x)];
                }
              }
            }
          }
        }

        if (m_algorithm == MSlic)
        {
          float ssumch = 0.0f;
          for (int b = 0; b < channelsNo; b++)
          {
            curch[b] = seedsC[b][labels.at<int>(j, k)];
            // squared distance
            float diff = curch[b] - adjch[b];
            ssumch += diff * diff;
          }
          // L2 distance with adj
          diffch = sqrt(ssumch);
          adaptk.push_back(m_adaptk[currentlabel]);
        }

        int count(1);
        for (int c = 0; c < count; c++)
        {
          for (int n = 0; n < 4; n++)
          {
            int x = xvec[c] + dx4[n];
            int y = yvec[c] + dy4[n];

            if ((x >= 0 && x < widthImg) && (y >= 0 && y < heightImg))
            {
              if (INT_MAX == nlabels.at<int>(y, x) &&
                  labels.at<int>(j, k) == labels.at<int>(y, x))
              {
                xvec[count] = x;
                yvec[count] = y;
                nlabels.at<int>(y, x) = label;
                count++;
              }
            }
          }
        }
        // MSlic only
        if (m_algorithm == MSlic)
        {
          if (m_cur_iter < m_iterations - 1)
          {
            hashtable[label] = count;
            //-------------------------------------------------------
            // If segment size is less then a limit, or is very similar
            // to it's neighbour assign adjacent label found before,
            // and decrement label count.
            //-------------------------------------------------------
            if ((count <= min_sp_sz) ||
                ((diffch < m_merge) &&
                 (hashtable[adjlabel] + hashtable[(int)adaptk.size() - 1] <= 3 * regionLength * regionLength)))
            {
              if ((diffch < m_merge) &&
                  (hashtable[adjlabel] + hashtable[(int)adaptk.size() - 1] <= 3 * regionLength * regionLength))
              {
                adaptk[adjlabel] = min(2.0f, float(adaptk[adjlabel] + adaptk[(int)adaptk.size() - 1]));
                hashtable[adjlabel] += hashtable[(int)adaptk.size() - 1];
              }

              for (int c = 0; c < count; c++)
              {
                nlabels.at<int>(yvec[c], xvec[c]) = adjlabel;
              }

              label--;
              adaptk.pop_back();
            }
          }
          else
          {
            //-------------------------------------------------------
            // If segment size is less then a limit, assign an
            // adjacent label found before, and decrement label count.
            //-------------------------------------------------------
            if (count <= min_sp_sz)
            {
              for (int c = 0; c < count; c++)
              {
                nlabels.at<int>(yvec[c], xvec[c]) = adjlabel;
              }
              label--;
            }
          }
          // Slic or Slico
        }
        else
        {
          //-------------------------------------------------------
          // If segment size is less then a limit, assign an
          // adjacent label found before, and decrement label count.
          //-------------------------------------------------------
          if (count <= min_sp_sz)
          {
            for (int c = 0; c < count; c++)
            {
              nlabels.at<int>(yvec[c], xvec[c]) = adjlabel;
            }
            label--;
          }
        }
        label++;
      }
    }
  }
  // replace old
  labels = nlabels;
  kSuperpixels = label;

  m_adaptk.clear();
  m_adaptk = adaptk;
}

/*
 * DetectChEdges
 */
inline void SlicImpl::DetectChEdges(Mat &edgemag)
{
  Mat dx, dy;
  Mat S_dx, S_dy;

  for (int c = 0; c < channelsNo; c++)
  {
    // derivate
    Sobel(channels[c], dx, CV_32F, 1, 0, 1, 1.0f, 0.0f, BORDER_DEFAULT);
    Sobel(channels[c], dy, CV_32F, 0, 1, 1, 1.0f, 0.0f, BORDER_DEFAULT);

    // acumulate ^2 derivate
    S_dx = S_dx + dx.mul(dx);
    S_dy = S_dy + dy.mul(dy);
  }
  // total magnitude
  edgemag += S_dx + S_dy;
}

/*
 * PerturbSeeds
 */
inline void SlicImpl::PerturbSeeds(const Mat &edgemag)
{
  const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
  const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};

  for (int n = 0; n < kSuperpixels; n++)
  {
    int ox = (int)seedsX[n]; //original x
    int oy = (int)seedsY[n]; //original y

    int storex = ox;
    int storey = oy;
    for (int i = 0; i < 8; i++)
    {
      int nx = ox + dx8[i]; //new x
      int ny = oy + dy8[i]; //new y

      if (nx >= 0 && nx < widthImg && ny >= 0 && ny < heightImg)
      {
        if (edgemag.at<float>(ny, nx) < edgemag.at<float>(storey, storex))
        {
          storex = nx;
          storey = ny;
        }
      }
    }
    if (storex != ox && storey != oy)
    {
      seedsX[n] = (float)storex;
      seedsY[n] = (float)storey;

      switch (channels[0].depth())
      {
      case CV_8U:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<uchar>(storey, storex);
        break;

      case CV_8S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<char>(storey, storex);
        break;

      case CV_16U:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<ushort>(storey, storex);
        break;

      case CV_16S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<short>(storey, storex);
        break;

      case CV_32S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = (float)channels[b].at<int>(storey, storex);
        break;

      case CV_32F:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<float>(storey, storex);
        break;

      case CV_64F:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = (float)channels[b].at<double>(storey, storex);
        break;

      default:
        CV_Error(Error::StsInternal, "Invalid matrix depth");
        break;
      }
    }
  }
}

/*
 * GetChannelsSeeds_ForGivenStepSize
 *
 * The k seed values are
 * taken as uniform spatial
 * pixel samples.
 *
 */
inline void SlicImpl::GetChSeedsS()
{
  int n = 0;
  int numseeds = 0;

  int xstrips = int(0.5f + float(widthImg) / float(regionLength));
  int ystrips = int(0.5f + float(heightImg) / float(regionLength));

  int xerr = widthImg - regionLength * xstrips;
  int yerr = heightImg - regionLength * ystrips;

  float xerrperstrip = float(xerr) / float(xstrips);
  float yerrperstrip = float(yerr) / float(ystrips);

  int xoff = regionLength / 2;
  int yoff = regionLength / 2;

  numseeds = xstrips * ystrips;

  for (int b = 0; b < channelsNo; b++)
    seedsC[b].resize(numseeds);

  seedsX.resize(numseeds);
  seedsY.resize(numseeds);

  for (int y = 0; y < ystrips; y++)
  {
    int ye = y * (int)yerrperstrip;
    int Y = y * regionLength + yoff + ye;
    if (Y > heightImg - 1)
      continue;
    for (int x = 0; x < xstrips; x++)
    {
      int xe = x * (int)xerrperstrip;
      int X = x * regionLength + xoff + xe;
      if (X > widthImg - 1)
        continue;

      switch (channels[0].depth())
      {
      case CV_8U:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<uchar>(Y, X);
        break;

      case CV_8S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<char>(Y, X);
        break;

      case CV_16U:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<ushort>(Y, X);
        break;

      case CV_16S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<short>(Y, X);
        break;

      case CV_32S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = (float)channels[b].at<int>(Y, X);
        break;

      case CV_32F:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = channels[b].at<float>(Y, X);
        break;

      case CV_64F:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b][n] = (float)channels[b].at<double>(Y, X);
        break;

      default:
        CV_Error(Error::StsInternal, "Invalid matrix depth");
        break;
      }

      seedsX[n] = (float)X;
      seedsY[n] = (float)Y;

      n++;
    }
  }
}

/*
 * GetChannlesSeeds_ForGivenK
 *
 * The k seed values are
 * taken as uniform spatial
 * pixel samples.
 *
 */
inline void SlicImpl::GetChSeedsK()
{
  int xoff = regionLength / 2;
  int yoff = regionLength / 2;
  int n = 0;
  int r = 0;
  for (int y = 0; y < heightImg; y++)
  {
    int Y = y * regionLength + yoff;
    if (Y > heightImg - 1)
      continue;
    for (int x = 0; x < widthImg; x++)
    {
      // hex grid
      int X = x * regionLength + (xoff << (r & 0x1));
      if (X > widthImg - 1)
        continue;

      switch (channels[0].depth())
      {
      case CV_8U:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b].push_back(channels[b].at<uchar>(Y, X));
        break;

      case CV_8S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b].push_back(channels[b].at<char>(Y, X));
        break;

      case CV_16U:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b].push_back(channels[b].at<ushort>(Y, X));
        break;

      case CV_16S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b].push_back(channels[b].at<short>(Y, X));
        break;

      case CV_32S:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b].push_back((float)channels[b].at<int>(Y, X));
        break;

      case CV_32F:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b].push_back(channels[b].at<float>(Y, X));
        break;

      case CV_64F:
        for (int b = 0; b < channelsNo; b++)
          seedsC[b].push_back((float)channels[b].at<double>(Y, X));
        break;

      default:
        CV_Error(Error::StsInternal, "Invalid matrix depth");
        break;
      }

      seedsX.push_back((float)X);
      seedsY.push_back((float)Y);

      n++;
    }
    r++;
  }
}

struct SeedNormInvoker : ParallelLoopBody
{
  SeedNormInvoker(vector<vector<float> > *_kseeds, vector<vector<float> > *_sigma,
                  vector<int> *_clustersize, vector<float> *_sigmax, vector<float> *_sigmay,
                  vector<float> *_kseedsx, vector<float> *_kseedsy, int _nr_channels)
  {
    sigma = _sigma;
    kseeds = _kseeds;
    sigmax = _sigmax;
    sigmay = _sigmay;
    kseedsx = _kseedsx;
    kseedsy = _kseedsy;
    nr_channels = _nr_channels;
    clustersize = _clustersize;
  }

  void operator()(const cv::Range &range) const
  {
    for (int k = range.start; k < range.end; ++k)
    {
      // if (clustersize->at(k) <= 0)
      //   clustersize->at(k) = 1;

      if (clustersize->at(k) < 0){
        cout<<"error"<<endl;
        exit(1);
      }

      if (clustersize->at(k) == 0){
        clustersize->at(k) = 1;
        //exit(1);
      }
        

      for (int b = 0; b < nr_channels; b++)
        kseeds->at(b)[k] = sigma->at(b)[k] / float(clustersize->at(k));

      kseedsx->at(k) = sigmax->at(k) / float(clustersize->at(k));
      kseedsy->at(k) = sigmay->at(k) / float(clustersize->at(k));
    } // end for k
  }
  vector<float> *sigmax;
  vector<float> *sigmay;
  vector<float> *kseedsx;
  vector<float> *kseedsy;
  vector<int> *clustersize;
  vector<vector<float> > *sigma;
  vector<vector<float> > *kseeds;
  int nr_channels;
};

struct SeedsCenters
{
  SeedsCenters(const vector<Mat> &_chvec, const Mat &_klabels,
               const int _numlabels, const int _nr_channels)
  {
    chvec = _chvec;
    klabels = _klabels;
    numlabels = _numlabels;
    nr_channels = _nr_channels;

    // allocate and init arrays
    sigma.resize(nr_channels);
    for (int b = 0; b < nr_channels; b++)
      sigma[b].assign(numlabels, 0);

    sigmax.assign(numlabels, 0);
    sigmay.assign(numlabels, 0);
    clustersize.assign(numlabels, 0);
  }

  SeedsCenters(const SeedsCenters &counter, Split)
  {
    *this = counter;
    // refill with zero all arrays
    for (int b = 0; b < nr_channels; b++)
      fill(sigma[b].begin(), sigma[b].end(), 0.0f);

    fill(sigmax.begin(), sigmax.end(), 0.0f);
    fill(sigmay.begin(), sigmay.end(), 0.0f);
    fill(clustersize.begin(), clustersize.end(), 0);
  }

  void operator()(const BlockedRange &range)
  {
    // previous block state
    vector<float> tmp_sigmax = sigmax;
    vector<float> tmp_sigmay = sigmay;
    vector<vector<float> > tmp_sigma = sigma;
    vector<int> tmp_clustersize = clustersize;

    for (int x = range.begin(); x != range.end(); x++)
    {
      for (int y = 0; y < chvec[0].rows; y++)
      {
        int idx = klabels.at<int>(y, x);

        switch (chvec[0].depth())
        {
        case CV_8U:
          for (int b = 0; b < nr_channels; b++)
            tmp_sigma[b][idx] += chvec[b].at<uchar>(y, x);
          break;

        case CV_8S:
          for (int b = 0; b < nr_channels; b++)
            tmp_sigma[b][idx] += chvec[b].at<char>(y, x);
          break;

        case CV_16U:
          for (int b = 0; b < nr_channels; b++)
            tmp_sigma[b][idx] += chvec[b].at<ushort>(y, x);
          break;

        case CV_16S:
          for (int b = 0; b < nr_channels; b++)
            tmp_sigma[b][idx] += chvec[b].at<short>(y, x);
          break;

        case CV_32S:
          for (int b = 0; b < nr_channels; b++)
            tmp_sigma[b][idx] += chvec[b].at<int>(y, x);
          break;

        case CV_32F:
          for (int b = 0; b < nr_channels; b++)
            tmp_sigma[b][idx] += chvec[b].at<float>(y, x);
          break;

        case CV_64F:
          for (int b = 0; b < nr_channels; b++)
            tmp_sigma[b][idx] += (float)chvec[b].at<double>(y, x);
          break;

        default:
          CV_Error(Error::StsInternal, "Invalid matrix depth");
          break;
        }

        tmp_sigmax[idx] += x;
        tmp_sigmay[idx] += y;

        tmp_clustersize[idx]++;
      }
    }
    sigma = tmp_sigma;
    sigmax = tmp_sigmax;
    sigmay = tmp_sigmay;
    clustersize = tmp_clustersize;
  }

  void join(SeedsCenters &sc)
  {
    for (int l = 0; l < numlabels; l++)
    {
      sigmax[l] += sc.sigmax[l];
      sigmay[l] += sc.sigmay[l];
      for (int b = 0; b < nr_channels; b++)
        sigma[b][l] += sc.sigma[b][l];
      clustersize[l] += sc.clustersize[l];
    }
  }

  Mat klabels;
  int numlabels;
  int nr_channels;
  vector<Mat> chvec;
  vector<float> sigmax;
  vector<float> sigmay;
  vector<int> clustersize;
  vector<vector<float> > sigma;
};

struct SlicoGrowInvoker : ParallelLoopBody
{
  SlicoGrowInvoker(vector<Mat> *_chvec, Mat *_distchans, Mat *_distxy, Mat *_distvec,
                   Mat *_klabels, float _kseedsxn, float _kseedsyn, float _xywt,
                   float _maxchansn, vector<vector<float> > *_kseeds,
                   int _x1, int _x2, int _nr_channels, int _n)
  {
    chvec = _chvec;
    distchans = _distchans;
    distxy = _distxy;
    distvec = _distvec;
    kseedsxn = _kseedsxn;
    kseedsyn = _kseedsyn;
    klabels = _klabels;
    maxchansn = _maxchansn;
    kseeds = _kseeds;
    x1 = _x1;
    x2 = _x2;
    n = _n;
    xywt = _xywt;
    nr_channels = _nr_channels;
  }

  void operator()(const cv::Range &range) const
  {
    int cols = klabels->cols;
    int rows = klabels->rows;
    for (int y = range.start; y < range.end; ++y)
    {
      for (int x = x1; x < x2; x++)
      {
        CV_Assert(y < rows && x < cols && y >= 0 && x >= 0);
        distchans->at<float>(y, x) = 0;

        switch (chvec->at(0).depth())
        {
        case CV_8U:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<uchar>(y, x) - kseeds->at(b)[n];
            distchans->at<float>(y, x) += diff * diff;
          }
          break;

        case CV_8S:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<char>(y, x) - kseeds->at(b)[n];
            distchans->at<float>(y, x) += diff * diff;
          }
          break;

        case CV_16U:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<ushort>(y, x) - kseeds->at(b)[n];
            distchans->at<float>(y, x) += diff * diff;
          }
          break;

        case CV_16S:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<short>(y, x) - kseeds->at(b)[n];
            distchans->at<float>(y, x) += diff * diff;
          }
          break;

        case CV_32S:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<int>(y, x) - kseeds->at(b)[n];
            distchans->at<float>(y, x) += diff * diff;
          }
          break;

        case CV_32F:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<float>(y, x) - kseeds->at(b)[n];
            distchans->at<float>(y, x) += diff * diff;
          }
          break;

        case CV_64F:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = float(chvec->at(b).at<double>(y, x) - kseeds->at(b)[n]);
            distchans->at<float>(y, x) += diff * diff;
          }
          break;

        default:
          CV_Error(Error::StsInternal, "Invalid matrix depth");
          break;
        }

        float difx = x - kseedsxn;
        float dify = y - kseedsyn;
        distxy->at<float>(y, x) = difx * difx + dify * dify;

        // only varying m, prettier superpixels
        float dist = distchans->at<float>(y, x) / maxchansn + distxy->at<float>(y, x) / xywt;

        if (dist < distvec->at<float>(y, x))
        {
          distvec->at<float>(y, x) = dist;
          klabels->at<int>(y, x) = n;
        }
      } // end for x
    }   // end for y
  }

  Mat *klabels;
  vector<vector<float> > *kseeds;
  float maxchansn, xywt;
  vector<Mat> *chvec;
  Mat *distchans, *distxy, *distvec;
  float kseedsxn, kseedsyn;
  int x1, x2, nr_channels, n;
};

/*
 *
 *    Magic Slic - no parameters
 *
 *    Performs k mean segmentation. It is fast because it looks locally, not
 * over the entire image.
 * This function picks the maximum value of color distance as compact factor
 * M and maximum pixel distance as grid step size S from each cluster (13 April 2011).
 * So no need to input a constant value of M and S. There are two clear
 * advantages:
 *
 * [1] The algorithm now better handles both textured and non-textured regions
 * [2] There is not need to set any parameters!!!
 *
 * Slico (or Slic Zero) dynamically varies only the compactness factor S,
 * not the step size S.
 *
 */
inline void SlicImpl::PerformSlico(const int &itrnum)
{
  Mat distxy(heightImg, widthImg, CV_32F, Scalar::all(FLT_MAX));
  Mat distvec(heightImg, widthImg, CV_32F, Scalar::all(FLT_MAX));
  Mat distchans(heightImg, widthImg, CV_32F, Scalar::all(FLT_MAX));

  // this is the variable value of M, just start with 10
  vector<float> maxchans(kSuperpixels, FLT_MIN);
  // this is the variable value of M, just start with 10
  vector<float> maxxy(kSuperpixels, FLT_MIN);
  // note: this is different from how usual Slic/LKM works
  const float xywt = float(regionLength * regionLength);

  for (int itr = 0; itr < itrnum; itr++)
  {
    distvec.setTo(FLT_MAX);
    for (int n = 0; n < kSuperpixels; n++)
    {
      int y1 = max(0, (int)seedsY[n] - regionLength);
      int y2 = min(heightImg, (int)seedsY[n] + regionLength);
      int x1 = max(0, (int)seedsX[n] - regionLength);
      int x2 = min((int)widthImg, (int)seedsX[n] + regionLength);

      parallel_for_(Range(y1, y2), SlicoGrowInvoker(&channels, &distchans, &distxy, &distvec,
                                                    &labels, seedsX[n], seedsY[n], xywt, maxchans[n], &seedsC,
                                                    x1, x2, channelsNo, n));
    }
    //-----------------------------------------------------------------
    // Assign the max color distance for a cluster
    //-----------------------------------------------------------------
    if (itr == 0)
    {
      maxchans.assign(kSuperpixels, FLT_MIN);
      maxxy.assign(kSuperpixels, FLT_MIN);
    }

    for (int x = 0; x < widthImg; x++)
    {
      for (int y = 0; y < heightImg; y++)
      {
        int idx = labels.at<int>(y, x);

        if (maxchans[idx] < distchans.at<float>(y, x))
          maxchans[idx] = distchans.at<float>(y, x);

        if (maxxy[idx] < distxy.at<float>(y, x))
          maxxy[idx] = distxy.at<float>(y, x);
      }
    }
    //-----------------------------------------------------------------
    // Recalculate the centroid and store in the seed values
    //-----------------------------------------------------------------

    // parallel reduce structure
    SeedsCenters sc(channels, labels, kSuperpixels, channelsNo);

    // accumulate center distances
    parallel_reduce(BlockedRange(0, widthImg), sc);

    // normalize centers
    parallel_for_(Range(0, kSuperpixels), SeedNormInvoker(&seedsC, &sc.sigma,
                                                          &sc.clustersize, &sc.sigmax, &sc.sigmay, &seedsX, &seedsY, channelsNo));
  }
}

struct SlicGrowInvoker : ParallelLoopBody
{
  SlicGrowInvoker(vector<Mat> *_chvec, Mat *_distvec, Mat *_klabels,
                  float _kseedsxn, float _kseedsyn, float _xywt,
                  vector<vector<float> > *_kseeds, int _x1, int _x2,
                  int _nr_channels, int _n)
  {
    chvec = _chvec;
    distvec = _distvec;
    kseedsxn = _kseedsxn;
    kseedsyn = _kseedsyn;
    klabels = _klabels;
    kseeds = _kseeds;
    x1 = _x1;
    x2 = _x2;
    n = _n;
    xywt = _xywt;
    nr_channels = _nr_channels;
  }

  void operator()(const cv::Range &range) const
  {
    for (int y = range.start; y < range.end; ++y)
    {
      for (int x = x1; x < x2; x++)
      {
        float dist = 0;

        switch (chvec->at(0).depth())
        {
        case CV_8U:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<uchar>(y, x) - kseeds->at(b)[n];
            dist += diff * diff;
          }
          break;

        case CV_8S:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<char>(y, x) - kseeds->at(b)[n];
            dist += diff * diff;
          }
          break;

        case CV_16U:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<ushort>(y, x) - kseeds->at(b)[n];
            dist += diff * diff;
          }
          break;

        case CV_16S:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<short>(y, x) - kseeds->at(b)[n];
            dist += diff * diff;
          }
          break;

        case CV_32S:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<int>(y, x) - kseeds->at(b)[n];
            dist += diff * diff;
          }
          break;

        case CV_32F:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = chvec->at(b).at<float>(y, x) - kseeds->at(b)[n];
            dist += diff * diff;
          }
          break;

        case CV_64F:
          for (int b = 0; b < nr_channels; b++)
          {
            float diff = float(chvec->at(b).at<double>(y, x) - kseeds->at(b)[n]);
            dist += diff * diff;
          }
          break;

        default:
          CV_Error(Error::StsInternal, "Invalid matrix depth");
          break;
        }

        float difx = x - kseedsxn;
        float dify = y - kseedsyn;
        float distxy = difx * difx + dify * dify;

        //dist += distxy / xywt;

        //this would be more exact but expensive
        dist = sqrt(dist) + sqrt(distxy / xywt);

        if (dist < distvec->at<float>(y, x))
        {
          distvec->at<float>(y, x) = dist;
          klabels->at<int>(y, x) = n;
        }
      } //end for x
    }   // end for y
  }

  Mat *klabels;
  vector<vector<float> > *kseeds;
  float xywt;
  vector<Mat> *chvec;
  Mat *distvec;
  float kseedsxn, kseedsyn;
  int x1, x2, nr_channels, n;
};

/*
 *    PerformSuperpixelSlic
 *
 *    Performs k mean segmentation. It is fast because it looks locally, not
 * over the entire image.
 *
 */
inline void SlicImpl::PerformSlic(const int &itrnum)
{
  Mat distvec(heightImg, widthImg, CV_32F);

  const float xywt = (regionLength / compactness) * (regionLength / compactness);

  for (int itr = 0; itr < itrnum; itr++)
  {
    distvec.setTo(FLT_MAX);
    for (int n = 0; n < kSuperpixels; n++)
    {
      int y1 = max(0, (int)seedsY[n] - regionLength);
      int y2 = min(heightImg, (int)seedsY[n] + regionLength);
      int x1 = max(0, (int)seedsX[n] - regionLength);
      int x2 = min((int)widthImg, (int)seedsX[n] + regionLength);

      parallel_for_(Range(y1, y2), SlicGrowInvoker(&channels, &distvec,
                                                   &labels, seedsX[n], seedsY[n], xywt, &seedsC,
                                                   x1, x2, channelsNo, n));
    }

    //-----------------------------------------------------------------
    // Recalculate the centroid and store in the seed values
    //-----------------------------------------------------------------
    // instead of reassigning memory on each iteration, just reset.

    // parallel reduce structure
    SeedsCenters sc(channels, labels, kSuperpixels, channelsNo);

    // accumulate center distances
    parallel_reduce(BlockedRange(0, widthImg), sc);

    // normalize centers
    parallel_for_(Range(0, kSuperpixels), SeedNormInvoker(&seedsC, &sc.sigma,
                                                          &sc.clustersize, &sc.sigmax, &sc.sigmay, &seedsX, &seedsY, channelsNo));
  }
}

/*
 *    PerformSuperpixelMSlic
 *
 *
 */
inline void SlicImpl::PerformMSlic(const int &itrnum)
{
  vector<vector<float> > sigma(channelsNo);
  for (int b = 0; b < channelsNo; b++)
    sigma[b].resize(kSuperpixels, 0);

  Mat distvec(heightImg, widthImg, CV_32F);

  const float xywt = (regionLength / compactness) * (regionLength / compactness);

  int offset = regionLength;

  // from paper
  m_split = 4.0f;
  m_ratio = 5.0f;

  for (int itr = 0; itr < itrnum; itr++)
  {
    m_cur_iter = itr;

    distvec.setTo(FLT_MAX);
    for (int n = 0; n < kSuperpixels; n++)
    {
      if (m_adaptk[n] < 1.0f)
        offset = int(regionLength * m_adaptk[n]);
      else
        offset = int(regionLength * m_adaptk[n]);

      int y1 = max(0, (int)seedsY[n] - offset);
      int y2 = min(heightImg, (int)seedsY[n] + offset);
      int x1 = max(0, (int)seedsX[n] - offset);
      int x2 = min(widthImg, (int)seedsX[n] + offset);

      parallel_for_(Range(y1, y2), SlicGrowInvoker(&channels, &distvec,
                                                   &labels, seedsX[n], seedsY[n], xywt, &seedsC,
                                                   x1, x2, channelsNo, n));
    }

    //-----------------------------------------------------------------
    // Recalculate the centroid and store in the seed values
    //-----------------------------------------------------------------
    // instead of reassigning memory on each iteration, just reset.

    // parallel reduce structure
    SeedsCenters sc(channels, labels, kSuperpixels, channelsNo);

    // accumulate center distances
    parallel_reduce(BlockedRange(0, widthImg), sc);

    // normalize centers
    parallel_for_(Range(0, kSuperpixels), SeedNormInvoker(&seedsC, &sc.sigma,
                                                          &sc.clustersize, &sc.sigmax, &sc.sigmay, &seedsX, &seedsY, channelsNo));

    // 13% as in original paper
    enforceLabelConnectivity(13);
    SuperpixelSplit();
  }
}

inline void SlicImpl::SuperpixelSplit()
{
  Mat klabels = labels.clone();

  // parallel reduce structure
  SeedsCenters msc(channels, labels, kSuperpixels, channelsNo);

  // accumulate center distances
  parallel_reduce(BlockedRange(0, widthImg), msc);

  const float invwt = 1.0f / ((regionLength / compactness) * (regionLength / compactness));
  const float sqrt_invwt = sqrt(invwt);

  if (m_cur_iter < m_iterations - 2)
  {
    vector<float> avglabs(kSuperpixels, 0);
    for (int y = 0; y < heightImg - 1; y++)
    {
      for (int x = 0; x < widthImg - 1; x++)
      {
        if (klabels.at<int>(y, x) == klabels.at<int>(y + 1, x) &&
            klabels.at<int>(y, x) == klabels.at<int>(y, x + 1))
        {
          float x1 = 1, y1 = 0;
          float x2 = 0, y2 = 1;

          vector<float> ch1(channelsNo);
          vector<float> ch2(channelsNo);

          switch (channels.at(0).depth())
          {
          case CV_8U:
            for (int c = 0; c < channelsNo; c++)
            {
              ch1[c] = float(channels[c].at<uchar>(y + 1, x) - channels[c].at<uchar>(y, x));
              ch2[c] = float(channels[c].at<uchar>(y, x + 1) - channels[c].at<uchar>(y, x));

              ch1[c] /= sqrt_invwt;
              ch2[c] /= sqrt_invwt;
            }
            break;

          case CV_8S:
            for (int c = 0; c < channelsNo; c++)
            {
              ch1[c] = float(channels[c].at<char>(y + 1, x) - channels[c].at<char>(y, x));
              ch2[c] = float(channels[c].at<char>(y, x + 1) - channels[c].at<char>(y, x));

              ch1[c] /= sqrt_invwt;
              ch2[c] /= sqrt_invwt;
            }
            break;

          case CV_16U:
            for (int c = 0; c < channelsNo; c++)
            {
              ch1[c] = float(channels[c].at<ushort>(y + 1, x) - channels[c].at<ushort>(y, x));
              ch2[c] = float(channels[c].at<ushort>(y, x + 1) - channels[c].at<ushort>(y, x));

              ch1[c] /= sqrt_invwt;
              ch2[c] /= sqrt_invwt;
            }
            break;

          case CV_16S:
            for (int c = 0; c < channelsNo; c++)
            {
              ch1[c] = float(channels[c].at<short>(y + 1, x) - channels[c].at<short>(y, x));
              ch2[c] = float(channels[c].at<short>(y, x + 1) - channels[c].at<short>(y, x));

              ch1[c] /= sqrt_invwt;
              ch2[c] /= sqrt_invwt;
            }
            break;

          case CV_32S:
            for (int c = 0; c < channelsNo; c++)
            {
              ch1[c] = float(channels[c].at<int>(y + 1, x) - channels[c].at<int>(y, x));
              ch2[c] = float(channels[c].at<int>(y, x + 1) - channels[c].at<int>(y, x));

              ch1[c] /= sqrt_invwt;
              ch2[c] /= sqrt_invwt;
            }
            break;

          case CV_32F:
            for (int c = 0; c < channelsNo; c++)
            {
              ch1[c] = channels[c].at<float>(y + 1, x) - channels[c].at<float>(y, x);
              ch2[c] = channels[c].at<float>(y, x + 1) - channels[c].at<float>(y, x);

              ch1[c] /= sqrt_invwt;
              ch2[c] /= sqrt_invwt;
            }
            break;

          case CV_64F:
            for (int c = 0; c < channelsNo; c++)
            {
              ch1[c] = float(channels[c].at<double>(y + 1, x) - channels[c].at<double>(y, x));
              ch2[c] = float(channels[c].at<double>(y, x + 1) - channels[c].at<double>(y, x));

              ch1[c] /= sqrt_invwt;
              ch2[c] /= sqrt_invwt;
            }
            break;

          default:
            CV_Error(Error::StsInternal, "Invalid matrix depth");
            break;
          }
          float ch11sqsum = 0.0f;
          float ch12sqsum = 0.0f;
          float ch22sqsum = 0.0f;
          for (int c = 0; c < channelsNo; c++)
          {
            ch11sqsum += ch1[c] * ch1[c];
            ch12sqsum += ch1[c] * ch2[c];
            ch22sqsum += ch2[c] * ch2[c];
          }

          // adjacent metric for N channels
          avglabs[klabels.at<int>(y, x)] += sqrt((x1 * x1 + y1 * y1 + ch11sqsum) * (x2 * x2 + y2 * y2 + ch22sqsum) - (x1 * x2 + y1 * y2 + ch12sqsum) * (x1 * x2 + y1 * y2 + ch12sqsum));
        }
      }
    }
    for (int i = 0; i < kSuperpixels; i++)
    {
      avglabs[i] /= regionLength * regionLength;
    }

    seedsX.clear();
    seedsY.clear();
    seedsX.resize(kSuperpixels, 0);
    seedsY.resize(kSuperpixels, 0);
    for (int c = 0; c < channelsNo; c++)
    {
      seedsC[c].clear();
      seedsC[c].resize(kSuperpixels, 0);
    }

    for (int k = 0; k < kSuperpixels; k++)
    {
      seedsX[k] = msc.sigmax[k] / msc.clustersize[k];
      seedsY[k] = msc.sigmay[k] / msc.clustersize[k];
      for (int c = 0; c < channelsNo; c++)
        seedsC[c][k] = msc.sigma[c][k] / msc.clustersize[k];
    }

    for (int k = 0; k < kSuperpixels; k++)
    {
      int xindex = 0, yindex = 0;
      if ((m_adaptk[k] <= 0.5f) ||
          (avglabs[k] < (m_split * m_ratio)))
      {
        seedsX[k] = msc.sigmax[k] / msc.clustersize[k];
        seedsY[k] = msc.sigmay[k] / msc.clustersize[k];
        for (int c = 0; c < channelsNo; c++)
          seedsC[c][k] = msc.sigma[c][k] / msc.clustersize[k];

        m_adaptk[k] = sqrt(m_ratio / avglabs[k]);
        m_adaptk[k] = max(0.5f, m_adaptk[k]);
        m_adaptk[k] = min(2.0f, m_adaptk[k]);
      }
      // if segment size is too large
      // split it and calculate four new seeds
      else
      {
        xindex = (int)(msc.sigmax[k] / msc.clustersize[k]);
        yindex = (int)(msc.sigmay[k] / msc.clustersize[k]);
        m_adaptk[k] = max(0.5f, m_adaptk[k] / 2);

        const float minadaptk = min(1.0f, m_adaptk[k]) * regionLength / 2;

        int x1 = (int)(xindex - minadaptk);
        int x2 = (int)(xindex + minadaptk);
        int x3 = (int)(xindex - minadaptk);
        int x4 = (int)(xindex + minadaptk);

        int y1 = (int)(yindex + minadaptk);
        int y2 = (int)(yindex + minadaptk);
        int y3 = (int)(yindex - minadaptk);
        int y4 = (int)(yindex - minadaptk);

        if (x1 < 0)
          x1 = 0;
        if (x2 >= widthImg)
          x2 = widthImg - 1;
        if (x3 < 0)
          x3 = 0;
        if (x4 >= widthImg)
          x4 = widthImg - 1;
        if (y1 >= heightImg)
          y1 = heightImg - 1;
        if (y2 >= heightImg)
          y2 = heightImg - 1;
        if (y3 < 0)
          y3 = 0;
        if (y4 < 0)
          y4 = 0;

        seedsX[k] = (float)x1;
        seedsY[k] = (float)y1;
        for (int c = 0; c < channelsNo; c++)
        {
          switch (channels[c].depth())
          {
          case CV_8U:
            seedsC[c][k] = channels[c].at<uchar>(y1, x1);
            break;

          case CV_8S:
            seedsC[c][k] = channels[c].at<char>(y1, x1);
            break;

          case CV_16U:
            seedsC[c][k] = channels[c].at<ushort>(y1, x1);
            break;

          case CV_16S:
            seedsC[c][k] = channels[c].at<short>(y1, x1);
            break;

          case CV_32S:
            seedsC[c][k] = float(channels[c].at<int>(y1, x1));
            break;

          case CV_32F:
            seedsC[c][k] = channels[c].at<float>(y1, x1);
            break;

          case CV_64F:
            seedsC[c][k] = float(channels[c].at<double>(y1, x1));
            break;

          default:
            CV_Error(Error::StsInternal, "Invalid matrix depth");
            break;
          }
        }

        seedsX.push_back((float)x2);
        seedsX.push_back((float)x3);
        seedsX.push_back((float)x4);
        seedsY.push_back((float)y2);
        seedsY.push_back((float)y3);
        seedsY.push_back((float)y4);

        for (int c = 0; c < channelsNo; c++)
        {
          switch (channels[c].depth())
          {
          case CV_8U:
            seedsC[c].push_back(channels[c].at<uchar>(y2, x2));
            seedsC[c].push_back(channels[c].at<uchar>(y3, x3));
            seedsC[c].push_back(channels[c].at<uchar>(y4, x4));
            break;

          case CV_8S:
            seedsC[c].push_back(channels[c].at<char>(y2, x2));
            seedsC[c].push_back(channels[c].at<char>(y3, x3));
            seedsC[c].push_back(channels[c].at<char>(y4, x4));
            break;

          case CV_16U:
            seedsC[c].push_back(channels[c].at<ushort>(y2, x2));
            seedsC[c].push_back(channels[c].at<ushort>(y3, x3));
            seedsC[c].push_back(channels[c].at<ushort>(y4, x4));
            break;

          case CV_16S:
            seedsC[c].push_back(channels[c].at<short>(y2, x2));
            seedsC[c].push_back(channels[c].at<short>(y3, x3));
            seedsC[c].push_back(channels[c].at<short>(y4, x4));
            break;

          case CV_32S:
            seedsC[c].push_back(float(channels[c].at<int>(y2, x2)));
            seedsC[c].push_back(float(channels[c].at<int>(y3, x3)));
            seedsC[c].push_back(float(channels[c].at<int>(y4, x4)));
            break;

          case CV_32F:
            seedsC[c].push_back(channels[c].at<float>(y2, x2));
            seedsC[c].push_back(channels[c].at<float>(y3, x3));
            seedsC[c].push_back(channels[c].at<float>(y4, x4));
            break;

          case CV_64F:
            seedsC[c].push_back(float(channels[c].at<double>(y2, x2)));
            seedsC[c].push_back(float(channels[c].at<double>(y3, x3)));
            seedsC[c].push_back(float(channels[c].at<double>(y4, x4)));
            break;

          default:
            CV_Error(Error::StsInternal, "Invalid matrix depth");
            break;
          }
        }
        m_adaptk.push_back(m_adaptk[k]);
        m_adaptk.push_back(m_adaptk[k]);
        m_adaptk.push_back(m_adaptk[k]);
        msc.clustersize.push_back(1);
        msc.clustersize.push_back(1);
        msc.clustersize.push_back(1);
      }
    }
  }
  else
  {
    seedsX.clear();
    seedsY.clear();
    seedsX.resize(kSuperpixels, 0);
    seedsY.resize(kSuperpixels, 0);
    for (int c = 0; c < channelsNo; c++)
    {
      seedsC[c].clear();
      seedsC[c].resize(kSuperpixels, 0);
    }

    for (int k = 0; k < kSuperpixels; k++)
    {
      seedsX[k] = msc.sigmax[k] / msc.clustersize[k];
      seedsY[k] = msc.sigmay[k] / msc.clustersize[k];
      for (int c = 0; c < channelsNo; c++)
        seedsC[c][k] = msc.sigma[c][k] / msc.clustersize[k];
    }
  }

  labels.release();
  labels = klabels.clone();

  // re-update amount of labels
  kSuperpixels = (int)seedsC[0].size();
}

} // namespace slicNora
} // namespace cv