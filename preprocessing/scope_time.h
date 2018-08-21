#pragma once

#include <boost/date_time/posix_time/posix_time.hpp>

class ScopeTime
{
protected:
  boost::posix_time::ptime start_time_;

public:
  inline ScopeTime (std::string title, bool debug) :
    title_ (title), debug_(debug)
  {
    start_time_ = boost::posix_time::microsec_clock::local_time ();
  }

  inline ScopeTime () :
    title_ (std::string (""))
  {
    start_time_ = boost::posix_time::microsec_clock::local_time ();
  }

  inline double
  getTime ()
  {
    boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
    return (static_cast<double> (((end_time - start_time_).total_milliseconds ())));
  }

  inline ~ScopeTime ()
  {
    double val = this->getTime ();
    if (debug_)
      std::cerr << title_ << " took " << val << "ms.\n";
  }

private:
  std::string title_;
  bool debug_;
};
