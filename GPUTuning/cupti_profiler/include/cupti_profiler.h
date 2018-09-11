#pragma once

#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <iostream>
#include <fstream>

#include <cupti.h>

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)


#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))



#ifdef DEBUG
  template<typename... Args>
  void _LOG(const char *msg, Args&&... args) {
    fprintf(stderr, "[Log]: ");
    fprintf(stderr, msg, args...);
    fprintf(stderr, "\n");
  }
  void _LOG(const char *msg) {
    fprintf(stderr, "[Log]: %s\n", msg);
  }
  template<typename... Args>
  void _DBG(const char *msg, Args&&... args) {
    fprintf(stderr, msg, args...);
  }
  void _DBG(const char *msg) {
    fprintf(stderr, "%s", msg);
  }
#else
  #define _LOG(...)
  #define _DBG(...)
#endif

namespace cupti_profiler 
{
  /*
   * @Abdullah, name space containing general utility functions
   * These functions do not specifically related to cupti or CUDA 
   */
  namespace util
  {

    /*
   * @Abdullah
   * Structure to hold the comparator function for const char*
   * It allows string comparison to be done when const char* is used as a key
   * in a map, usually if const char* is used as a key it uses address for
   * comparison
   */
    struct cmp_str
    {
       bool operator()(char const *a, char const *b)
       {
          return strcmp(a, b) < 0;
       }
    };
    
    /* 
     * @Summary, check if a file exists
     */
    bool fexists(std::string fileName) 
    {
      std::ifstream ifile(fileName);
      if (ifile)
      {
        ifile.close();
        return true;
      }
      else
      {
        return false;
      }
    }

  } //namespace util


  /* @Abdullah, Global variables for the cupti_profiler namespace*/
  static const char *dummy_kernel_name = "^^ DUMMY ^^";
  /* @Abdullah, Public static member, needed to save the execution time of each kernel */
  static std::map<const char*, double, util::cmp_str> kernel_exec_time = {{dummy_kernel_name, 0.0}};

  namespace detail 
  {

    // Pass-specific data
    struct pass_data_t {
      // the set of event groups to collect for a pass
      CUpti_EventGroupSet *event_groups;
      // the number of entries in eventIdArray and eventValueArray
      uint32_t num_events;
      // array of event ids
      std::vector<CUpti_EventID> event_ids;
      // array of event values
      std::vector<uint64_t> event_values;
    };

    struct kernel_data_t {
      typedef std::vector<uint64_t> event_val_t;
      typedef std::vector<CUpti_MetricValue> metric_val_t;

      kernel_data_t() : m_current_pass(0) {}

      std::vector<pass_data_t> m_pass_data;
      std::string m_name;

      int m_metric_passes;
      int m_event_passes;
      int m_current_pass;
      int m_total_passes;
      CUdevice m_device;

      /* @Abdullah, Execution time for the kernel, needed for some metrics*/
      // double m_exec_time;
      event_val_t m_event_values;
      metric_val_t m_metric_values;

    };


    void CUPTIAPI
    get_value_callback(void *userdata,
                       CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid,
                       const CUpti_CallbackData *cbInfo) 
    {

      // This callback is enabled only for launch so we shouldn't see
      // anything else.
      // @Abdullah, Added other CUDA version
      if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 && cbid!=CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
        fprintf(stderr, "%s:%d: Unexpected cbid %d\n", __FILE__, __LINE__, cbid);
        exit(-1);
      }

      const char *current_kernel_name = cbInfo->symbolName;

      _LOG("Current Kernel Name = %s", current_kernel_name);

      // Skip execution if kernel name is NULL string
      // TODO: Make sure this is fine
      if(!current_kernel_name) {
        _LOG("Empty kernel name string. Skipping...");
        return;
      }

      std::map<std::string, detail::kernel_data_t> *kernel_data =
        (std::map<std::string, detail::kernel_data_t> *)userdata;

      if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        // If this is kernel name hasn't been seen before
        if(kernel_data->count(current_kernel_name) == 0) {
          _LOG("New kernel encountered: %s", current_kernel_name);

          detail::kernel_data_t dummy =
            (*kernel_data)[dummy_kernel_name];
          detail::kernel_data_t k_data = dummy;

          k_data.m_name = current_kernel_name;

          auto& pass_data = k_data.m_pass_data;

          /* 
           * @Abdullah, collecting data in CUPTI_EVENT_COLLECTION_MODE_KERNEL
           * This means we are collecting data only for kernels
           * In other words, collection starts at kernel launch and ends once
           * the kernel is finished
           */
          CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                     CUPTI_EVENT_COLLECTION_MODE_KERNEL));

          for (int i = 0; i < pass_data[0].event_groups->numEventGroups; i++) {
            _LOG("  Enabling group %d", i);
            uint32_t all = 1;
            CUPTI_CALL(cuptiEventGroupSetAttribute(
                  pass_data[0].event_groups->eventGroups[i],
                  CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                  sizeof(all), &all));
            CUPTI_CALL(cuptiEventGroupEnable(
                  pass_data[0].event_groups->eventGroups[i]));

            (*kernel_data)[current_kernel_name] = k_data;
          }
        } else {
          auto& current_kernel = (*kernel_data)[current_kernel_name];
          auto const& pass_data = current_kernel.m_pass_data;

          int current_pass = current_kernel.m_current_pass;
          if(current_pass >= current_kernel.m_total_passes)
            return;

          _LOG("Current pass for %s: %d", current_kernel_name, current_pass);

          CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                CUPTI_EVENT_COLLECTION_MODE_KERNEL));

          for (int i = 0; i < pass_data[current_pass].event_groups->numEventGroups; i++) {
            _LOG("  Enabling group %d", i);
            uint32_t all = 1;
            CUPTI_CALL(cuptiEventGroupSetAttribute(
                  pass_data[current_pass].event_groups->eventGroups[i],
                  CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                  sizeof(all), &all));
            CUPTI_CALL(cuptiEventGroupEnable(
                  pass_data[current_pass].event_groups->eventGroups[i]));

          }
        }
      } else if(cbInfo->callbackSite == CUPTI_API_EXIT) {
        auto& current_kernel = (*kernel_data)[current_kernel_name];
        int current_pass = current_kernel.m_current_pass;

        if(current_pass >= current_kernel.m_total_passes)
          return;

        auto& pass_data =
          current_kernel.m_pass_data[current_pass];

        for (int i = 0; i < pass_data.event_groups->numEventGroups; i++) {
          CUpti_EventGroup group = pass_data.event_groups->eventGroups[i];
          CUpti_EventDomainID group_domain;
          uint32_t numEvents, numInstances, numTotalInstances;
          CUpti_EventID *eventIds;
          size_t groupDomainSize = sizeof(group_domain);
          size_t numEventsSize = sizeof(numEvents);
          size_t numInstancesSize = sizeof(numInstances);
          size_t numTotalInstancesSize = sizeof(numTotalInstances);
          uint64_t *values, normalized, sum;
          size_t valuesSize, eventIdsSize;

          CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                &groupDomainSize, &group_domain));
          CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
                current_kernel.m_device, group_domain,
                CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                &numTotalInstancesSize, &numTotalInstances));
          CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                &numInstancesSize, &numInstances));
          CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                &numEventsSize, &numEvents));
          eventIdsSize = numEvents * sizeof(CUpti_EventID);
          eventIds = (CUpti_EventID *)malloc(eventIdsSize);
          CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                CUPTI_EVENT_GROUP_ATTR_EVENTS,
                &eventIdsSize, eventIds));

          valuesSize = sizeof(uint64_t) * numInstances;
          values = (uint64_t *)malloc(valuesSize);

          for(int j = 0; j < numEvents; j++) {
            CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                       eventIds[j], &valuesSize, values));
            /*if (metric_data->eventIdx >= metric_data->numEvents) {
              fprintf(stderr, "[error]: Too many events collected, metric expects only %d\n",
                  (int)metric_data->numEvents);
              exit(-1);
            }*/

            // sum collect event values from all instances
            sum = 0;
            for(int k = 0; k < numInstances; k++)
              sum += values[k];

            // normalize the event value to represent the total number of
            // domain instances on the device
            normalized = (sum * numTotalInstances) / numInstances;

            pass_data.event_ids.push_back(eventIds[j]);
            pass_data.event_values.push_back(normalized);

            // print collected value
            {
              char eventName[128];
              size_t eventNameSize = sizeof(eventName) - 1;
              CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
                         &eventNameSize, eventName));
              eventName[127] = '\0';
              _DBG("\t%s = %llu (", eventName, (unsigned long long)sum);
              if (numInstances > 1) {
                for (int k = 0; k < numInstances; k++) {
                  if (k != 0)
                    _DBG(", ");
                  _DBG("%llu", (unsigned long long)values[k]);
                }
              }

              _DBG(")\n");
              _LOG("\t%s (normalized) (%llu * %u) / %u = %llu",
                  eventName, (unsigned long long)sum,
                  numTotalInstances, numInstances,
                  (unsigned long long)normalized);
            }
          }
          free(values);
        }

        for (int i = 0;
             i < pass_data.event_groups->numEventGroups;
             i++) {
          _LOG("  Disabling group %d", i);
          CUPTI_CALL(cuptiEventGroupDisable(
                     pass_data.event_groups->eventGroups[i]));
        }
        ++(*kernel_data)[current_kernel_name].m_current_pass;
      }
    }

    template<typename stream_t>
    void print_metric(CUpti_MetricID& id,
                      CUpti_MetricValue& value,
                      stream_t& s) 
    {
      CUpti_MetricValueKind value_kind;
      size_t value_kind_sz = sizeof(value_kind);
      CUPTI_CALL(cuptiMetricGetAttribute(id, CUPTI_METRIC_ATTR_VALUE_KIND,
                                         &value_kind_sz, &value_kind));
      switch(value_kind) {
      case CUPTI_METRIC_VALUE_KIND_DOUBLE:
        s << value.metricValueDouble;
        break;
      case CUPTI_METRIC_VALUE_KIND_UINT64:
        s << value.metricValueUint64;
        break;
      case CUPTI_METRIC_VALUE_KIND_INT64:
        s << value.metricValueInt64;
        break;
      case CUPTI_METRIC_VALUE_KIND_PERCENT:
        s << value.metricValuePercent;
        break;
      case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
        s << value.metricValueThroughput;
        break;
      case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
        s << value.metricValueUtilizationLevel;
        break;
      default:
        std::cerr << "[error]: unknown value kind\n";
        exit(-1);
      }
    }

  } // namespace detail



  struct profiler 
  {
    typedef std::vector<std::string> strvec_t;
    using event_val_t = detail::kernel_data_t::event_val_t;
    using metric_val_t = detail::kernel_data_t::metric_val_t;

    profiler(const strvec_t& events,
             const strvec_t& metrics,
             const int device_num = 0) :
      m_event_names(events),
      m_metric_names(metrics),
      m_device_num(device_num),
      m_num_metrics(metrics.size()),
      m_num_events(events.size()),
      m_metric_passes(0),
      m_event_passes(0) {

      int device_count = 0;

      // make sure activity is enabled before any CUDA API
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
      DRIVER_API_CALL(cuInit(0));
      DRIVER_API_CALL(cuDeviceGetCount(&device_count));
      if (device_count == 0) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        exit(1);
      }

      m_metric_ids.resize(m_num_metrics);
      m_event_ids.resize(m_num_events);

      // Init device, context and setup callback
      DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
      DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
      
      
      /* 
       * @Adbullah, need to collect duration of kernel execution without any event
       * collection enabled (some metrics need kernel duration as part of
       * calculation). The only accurate way to do this is by using the
       * activity API.
       */
      CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));


      CUPTI_CALL(cuptiSubscribe(&m_subscriber,
                 (CUpti_CallbackFunc)detail::get_value_callback,
                 &m_kernel_data));
      CUPTI_CALL(cuptiEnableCallback(1, m_subscriber,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
      /*@Abdullah, added callback for other cuda versions*/
      CUPTI_CALL(cuptiEnableCallback(1, m_subscriber,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

      CUpti_MetricID metric_ids[m_num_metrics];
      for(int i = 0; i < m_num_metrics; ++i) {
        CUPTI_CALL(cuptiMetricGetIdFromName(m_device,
                   m_metric_names[i].c_str(),
                   &metric_ids[i]));
      }
      CUpti_EventID event_ids[m_num_events];
      for(int i = 0; i < m_num_events; ++i) {
        CUPTI_CALL(cuptiEventGetIdFromName(m_device,
                   m_event_names[i].c_str(),
                   &event_ids[i]));
      }

      if(m_num_metrics > 0) {
        CUPTI_CALL(cuptiMetricCreateEventGroupSets(m_context,
                   sizeof(metric_ids), metric_ids,
                   &m_metric_pass_data));
        m_metric_passes = m_metric_pass_data->numSets;

        std::copy(metric_ids, metric_ids + m_num_metrics,
                  m_metric_ids.begin());
      }
      if(m_num_events > 0) {
        CUPTI_CALL(cuptiEventGroupSetsCreate(m_context,
                   sizeof(event_ids), event_ids,
                   &m_event_pass_data));
        m_event_passes = m_event_pass_data->numSets;

        std::copy(event_ids, event_ids + m_num_events,
                  m_event_ids.begin());
      }

      _LOG("# Metric Passes: %d\n", m_metric_passes);
      _LOG("# Event Passes: %d\n", m_event_passes);

      assert((m_metric_passes + m_event_passes) > 0);

      detail::kernel_data_t dummy_data;
      dummy_data.m_name = dummy_kernel_name;
      dummy_data.m_metric_passes = m_metric_passes;
      dummy_data.m_event_passes = m_event_passes;
      dummy_data.m_device = m_device;
      dummy_data.m_total_passes = m_metric_passes + m_event_passes;
      dummy_data.m_pass_data.resize(m_metric_passes + m_event_passes);

      auto& pass_data = dummy_data.m_pass_data;
      for(int i = 0; i < m_metric_passes; ++i) {
        int total_events = 0;
        _LOG("[metric] Looking at set (pass) %d", i);
        uint32_t num_events = 0;
        size_t num_events_size = sizeof(num_events);
        for(int j = 0; j < m_metric_pass_data->sets[i].numEventGroups; ++j) {
          CUPTI_CALL(cuptiEventGroupGetAttribute(
                m_metric_pass_data->sets[i].eventGroups[j],
                CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                &num_events_size, &num_events));
          _LOG("  Event Group %d, #Events = %d", j, num_events);
          total_events += num_events;
        }
        pass_data[i].event_groups = m_metric_pass_data->sets + i;
        pass_data[i].num_events = total_events;
      }

      for(int i = 0; i < m_event_passes; ++i) {
        int total_events = 0;
        _LOG("[event] Looking at set (pass) %d", i);
        uint32_t num_events = 0;
        size_t num_events_size = sizeof(num_events);
        for(int j = 0; j < m_event_pass_data->sets[i].numEventGroups; ++j) {
          CUPTI_CALL(cuptiEventGroupGetAttribute(
                m_event_pass_data->sets[i].eventGroups[j],
                CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                &num_events_size, &num_events));
          _LOG("  Event Group %d, #Events = %d", j, num_events);
          total_events += num_events;
        }
        pass_data[i + m_metric_passes].event_groups = m_event_pass_data->sets + i;
        pass_data[i + m_metric_passes].num_events = total_events;
      }

      m_kernel_data[dummy_kernel_name] = dummy_data;
    }

    ~profiler() {
    }

    int get_passes()
    { return m_metric_passes + m_event_passes; }

    /* 
     * @Abdullah,
     * Adding support for collecting kernel execution time
     * Some metrics require execution time to calculate the metric
     */

    /*
     * @Abdullah, 
     * @summary, bufferRequested function needed by the activity API
     */
    static void CUPTIAPI
    bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
    {
      uint8_t *rawBuffer;

      /* 
       * @Abdullah, why this number?? Related to Buffer size in a device?? 
       * Seems like this is related to the size of the record that is being returned
       * The record type is of struct CUpti_ActivityKernel4 
       */
      *size = 16 * 1024;
      rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

      /* 
       * @Abdullah, align the newly created buffer and point the requested buffer pointer
       * to the aligned address
       */
      *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
      *maxNumRecords = 0;

      if (*buffer == NULL) {
        printf("Error: out of memory\n");
        exit(-1);
      }
    }

    /*
     * @Abdullah, 
     * @summary, bufferCompleted function needed by the activity API
     */
    static void CUPTIAPI
    bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
    {
      CUptiResult status;
      CUpti_Activity *record = NULL;
      CUpti_ActivityKernel4 *kernel;

      /*@Abdullah, get the kernel records*/
      if (validSize > 0) 
      {
        do 
        {
          status = cuptiActivityGetNextRecord(buffer, validSize, &record);
          if (status == CUPTI_SUCCESS) 
          {
            // printActivity(record);
            kernel = (CUpti_ActivityKernel4 *)record;
            if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
              fprintf(stderr, "Error: expected kernel activity record, got %d\n", (int)kernel->kind);
              exit(-1);
            }

            const char *tmpkernelName = kernel->name;
            // std::string tmpkernelName = kernel->name;
            cupti_profiler::kernel_exec_time[tmpkernelName] = kernel->end - kernel->start;
            // profiler::m_kernel_data[tmpkernelName].m_exec_time = kernel->end - kernel->start;
            _LOG("Name of the kernel = %s, Execution time = %f", tmpkernelName, kernel_exec_time[kernel->name]);
            // kernelDuration = kernel->end - kernel->start;
          }
          else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
            break;
          else 
          {
            CUPTI_CALL(status);
          }
        } while (1);
        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) 
        {
          printf("Dropped %u activity records\n", (unsigned int) dropped);
        }
      }
      
      free(buffer);
    }
    /**************************/

    void start() 
    {
      /* 
       * @Adbullah, need to collect duration of kernel execution without any event
       * collection enabled (some metrics need kernel duration as part of
       * calculation). The only accurate way to do this is by using the
       * activity API.
       */
      // CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
      
      /* @Abdullah, 
       * calling this CUDA function so that the the CUDA environment
       * gets initialized, this solves the problem of not having the
       * CUDA environment initialized during the first time a kernel
       * is launched,
       * If a kernel launch is the first CUDA function that's called
       * Then we don't have any info on "const CUpti_CallbackData *cbInfo" struct
       * However, if we call a CUDA function before a call to the 
       * kernel it seems like the CUDA environment is initialized
       * And we have valid data in "cbInfo" struct in get_value_callback()
       *
       * @WARNING: This is a very crude way to do this
       * Ideally we would not want a cuda call inside a CUPTI library
       * Hence use it in the application
       * @SOLUTION: Add cudaDeviceSynchronize() before the first kernel invocation
       */
      // cudaDeviceSynchronize();

    }

    void stop() 
    {
      /*@Abdullah, Flush the activity data (execution time)*/
      CUPTI_CALL(cuptiActivityFlushAll(0));
      for(auto &k: m_kernel_data) 
      {
        auto& data = k.second.m_pass_data;

        if(k.first == dummy_kernel_name)
          continue;

        int total_events = 0;
        for(int i = 0; i < m_metric_passes; ++i) {
          //total_events += m_metric_data[i].num_events;
          total_events += data[i].num_events;
        }
        CUpti_MetricValue metric_value;
        CUpti_EventID *event_ids = new CUpti_EventID[total_events];
        uint64_t *event_values = new uint64_t[total_events];

        int running_sum = 0;
        for(int i = 0; i < m_metric_passes; ++i) {
          std::copy(data[i].event_ids.begin(),
                    data[i].event_ids.end(),
                    event_ids + running_sum);
          std::copy(data[i].event_values.begin(),
                    data[i].event_values.end(),
                    event_values + running_sum);
          running_sum += data[i].num_events;
        }

        for(int i = 0; i < m_num_metrics; ++i) 
        {

          /* 
           * @Test just checking if the issue of some metrics not being able to
           * compute was due to not having valid execution time value
           */
          const char *tmpkernelName = k.first.c_str();
          // _LOG("Execution time for kernel- %s is = %f", tmpkernelName, cupti_profiler::kernel_exec_time[tmpkernelName]);
          
          CUptiResult _status = cuptiMetricGetValue(m_device,
                     m_metric_ids[i],
                     total_events * sizeof(CUpti_EventID),
                     event_ids,
                     total_events * sizeof(uint64_t),
                     event_values,
                     cupti_profiler::kernel_exec_time[tmpkernelName], &metric_value);
          /*CUptiResult _status = cuptiMetricGetValue(m_device,
                     m_metric_ids[i],
                     total_events * sizeof(CUpti_EventID),
                     event_ids,
                     total_events * sizeof(uint64_t),
                     event_values,
                     0, &metric_value);*/
          if(_status != CUPTI_SUCCESS) {
            fprintf(stderr, "Metric value retrieval failed for metric %s\n",
                    m_metric_names[i].c_str());
            exit(-1);
          }
          k.second.m_metric_values.push_back(metric_value);
        }

        delete[] event_ids;
        delete[] event_values;

        std::map<CUpti_EventID, uint64_t> event_map;
        for(int i = m_metric_passes;
            i < (m_metric_passes + m_event_passes);
            ++i) {
          for(int j = 0; j < data[i].num_events; ++j) {
            event_map[data[i].event_ids[j]] =
              data[i].event_values[j];
          }
        }

        for(int i = 0; i < m_num_events; ++i) {
          k.second.m_event_values.push_back(
              event_map[m_event_ids[i]]);
        }
      }

      // Disable callback and unsubscribe
      CUPTI_CALL(cuptiEnableCallback(0, m_subscriber,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
      CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
    }




    template<typename stream>
    void print_event_values(stream& s,
                            bool print_names=true,
                            const char* kernel_separator = "; ") 
    {
      using ull_t = unsigned long long;


      /*fprintf(stderr, "Size of m_kernel_data =  %d\n", m_kernel_data.size());
      for (auto k: m_kernel_data)
      {
        fprintf(stderr, "Name of the kernel = %s\n", k.first);
        for (auto d:k.second.m_event_values)
        {
          fprintf(stderr, "%f\n", d);
        }
      }*/

      for(auto const& k: m_kernel_data) 
      {
        if(k.first == dummy_kernel_name)
          continue;
        fprintf(stderr,"--------------------------\n");
        fprintf(stderr,"%s: ",
              m_kernel_data[k.first].m_name.c_str());

        for(int i = 0; i < m_num_events; ++i) {
          printf("Event [%s] = %llu\n",
              m_event_names[i].c_str(),
              (ull_t)m_kernel_data[k.first].m_event_values[i]);
        }
        printf("\n");

        if(m_num_events <= 0)
          return;

        for(int i = 0; i < m_num_events; ++i) {
          if(print_names)
            s << "(" << m_event_names[i] << ","
              << (ull_t)m_kernel_data[k.first].m_event_values[i]
              << ") ";
          else
            s << (ull_t)m_kernel_data[k.first].m_event_values[i]
              << " ";
        }
        s << kernel_separator;
      }
      printf("\n");
    }

    /*template<typename stream>
    void print_metric_values(stream& s,
                             bool print_names=true,
                             const char* kernel_separator = "; ") 
    {
      if(m_num_metrics <= 0)
        return;

      for(auto const& k: m_kernel_data) {
        if(k.first == dummy_kernel_name)
          continue;

        //printf("%s: ",
        //       m_kernel_data[k.first].m_name.c_str());

        for(int i = 0; i < m_num_metrics; ++i) {
          if(print_names)
            s << "(" << m_metric_names[i] << ",";

          detail::print_metric(
            m_metric_ids[i],
            m_kernel_data[k.first].m_metric_values[i],
            s);

          if(print_names) s << ") ";
          else s << " ";
        }
        s << kernel_separator;
      }
      printf("\n");
    }*/


    /* @Abdullh
     * @summary, Modified version print_metric_values() to meet our need
     * We output the result in a different format
     */
    template<typename stream>
    void print_metric_values(stream& s,
                             std::string config) 
    {
      if(m_num_metrics <= 0)
        return;

      for(auto const& k: m_kernel_data) 
      {
        if(k.first == dummy_kernel_name)
          continue;
        for(int i = 0; i < m_num_metrics; ++i)
        {
          /*print kernel name*/
          s << k.first << ",";
          /*print metric name*/
          s << m_metric_names[i] << ",";
          s << config << ",";
          /*print actual data*/
          detail::print_metric(
            m_metric_ids[i],
            m_kernel_data[k.first].m_metric_values[i],
            s);
          s << "\n";
        }
      }
    }

    /*
     * @Abdullah
     * @Summary, a helper function that wirtes the header for the output file
     * @param, s: stream to write to
     *
     * Making it inline for potential better performance, given it does very
     * little work
     */
    template<typename stream>
    static void inline _print_output_header(stream& s)
    {
      s << "Kernel" << ",";
      s << "Metric" << ",";
      s << "Configuration" << ",";
      s << "Value" << "\n";
    }

    /*
     * @Abdullah, 
     * @Summary, Another wrapper to print print_metric_values()
     *
     * @param, fileName: Output fileName
     * @param, config: memory/resource configuration used for this kernel
     */
    // template<typename stream>
    void print_metric_values_to_file(std::string fileName,
                             std::string config)
    {
      std::ofstream ofs;
      if(util::fexists(fileName))
      {
        /*File exists, just append the records, no need for additional header*/
        ofs.open (fileName, std::ofstream::out | std::ofstream::app);
      }
      else
      {
        /*
         * Output file does not exist, create the output file and
         * write appropiate header 
         */
        ofs.open (fileName, std::ofstream::out | std::ofstream::app);
        _print_output_header(ofs);
      }
      print_metric_values(ofs, config);
      ofs.close();
    }

    template<typename stream>
    void print_events_and_metrics(stream& s,
                                  bool print_names = true,
                                  const char* kernel_separator = "; ") 
    {
      if(m_num_events <= 0 && m_num_metrics <= 0)
        return;

      using ull_t = unsigned long long;
      for(auto const& k: m_kernel_data) {
        if(k.first == dummy_kernel_name)
          continue;

        //printf("New kernel: %s \n",
        //       m_kernel_data[k.first].m_name.c_str());

        for(int i = 0; i < m_num_events; ++i) {
          if(print_names)
            s << "(" << m_event_names[i] << ","
              << (ull_t)m_kernel_data[k.first].m_event_values[i]
              << ") ";
          else
            s << (ull_t)m_kernel_data[k.first].m_event_values[i]
              << " ";
        }

        for(int i = 0; i < m_num_metrics; ++i) {
          if(print_names)
            s << "(" << m_metric_names[i] << ",";

          detail::print_metric(
            m_metric_ids[i],
            m_kernel_data[k.first].m_metric_values[i],
            s);

          if(print_names) s << ") ";
          else s << " ";
        }

        s << kernel_separator;
      }
      printf("\n");
    }

    std::vector<std::string> get_kernel_names() 
    {
      if(m_kernel_names.size() == 0) {
        for(auto const& k: m_kernel_data) {
          if(k.first == dummy_kernel_name)
            continue;
          m_kernel_names.push_back(k.first);
        }
      }
      return m_kernel_names;
    }

    event_val_t
    get_event_values(const char *kernel_name) 
    {
      if(m_num_events > 0)
        return m_kernel_data[kernel_name].m_event_values;
      else
        return event_val_t{};
    }

    metric_val_t get_metric_values(const char *kernel_name) 
    {
      if(m_num_metrics > 0)
        return m_kernel_data[kernel_name].m_metric_values;
      else
        return metric_val_t{};
    }

  
  
  private:
    int m_device_num;
    int m_num_metrics, m_num_events;
    const strvec_t& m_event_names;
    const strvec_t& m_metric_names;
    std::vector<CUpti_MetricID> m_metric_ids;
    std::vector<CUpti_EventID> m_event_ids;

    CUcontext m_context;
    CUdevice m_device;
    CUpti_SubscriberHandle m_subscriber;

    CUpti_EventGroupSets *m_metric_pass_data;
    CUpti_EventGroupSets *m_event_pass_data;

    int m_metric_passes, m_event_passes;
    // Kernel-specific (indexed by name) trace data
    std::map<std::string,
             detail::kernel_data_t> m_kernel_data;
    std::vector<std::string> m_kernel_names;

    int m_num_kernels;
  };
 
  
} // namespace cupti_profiler



