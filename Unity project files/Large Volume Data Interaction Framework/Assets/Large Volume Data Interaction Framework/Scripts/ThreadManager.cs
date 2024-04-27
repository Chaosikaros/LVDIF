using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading;

namespace ChaosIkaros
{
    public class ThreadManager : MonoBehaviour
    {
        public class DelayedAction
        {
            public float timeMarker = 0;
            public Action delayedAction = null;
            public DelayedAction(Action delayedAction, float timeMarker) {
                delayedAction = this.delayedAction;
                timeMarker = this.timeMarker;
            }
        }

        public static int maxThreads = 10;
        public static int threadCounter = 0;
        public static bool exist = false;
        public static ThreadManager threadManagerHolder;
        private List<Action> tempActions = new List<Action>();
        private List<Action> currentActions = new List<Action>();
        private List<DelayedAction> delayedActions = new List<DelayedAction>();
        private List<DelayedAction> currentDelayedActions = new List<DelayedAction>();
        public static ThreadManager threadManager
        {
            get
            {
                InitThreadManager();
                return threadManagerHolder;
            }
        }

        public static void InitThreadManager()
        {
            if (!exist)
            {
                threadCounter = 0;
                exist = true;
                GameObject managerHolder = new GameObject("ThreadManager");
                DontDestroyOnLoad(managerHolder);
                threadManagerHolder = managerHolder.AddComponent<ThreadManager>();
            }
        }

        public static void RunUnityAction(Action inputAction, float delayedTime = 0)
        {
            if (delayedTime != 0)
            {
                if (threadManager != null)
                    lock (threadManager.delayedActions)
                    {
                        threadManager.delayedActions.Add(new DelayedAction(
                             inputAction, delayedTime += Time.time
                        ));
                    }
            }
            else if (threadManager != null)
            {
                lock (threadManager.tempActions)
                {
                    threadManager.tempActions.Add(inputAction);
                }
            }
        }

        public static Thread RunOriginalAction(Action inputAction)
        {
            InitThreadManager();
            while (threadCounter >= maxThreads)
                Thread.Sleep(1);
            Interlocked.Increment(ref threadCounter);
            ThreadPool.QueueUserWorkItem(RunAction, inputAction);
            return null;
        }

        private static void RunAction(object action)
        {
            try
            {
                ((Action)action)();
            }
            catch
            {
            }
            Interlocked.Decrement(ref threadCounter);
        }

        private void FixedUpdate()
        {
            ThreadManagerLoop();
        }

        public void ThreadManagerLoop()
        {
            lock (tempActions)
            {
                currentActions.Clear();
                currentActions.AddRange(tempActions);
                tempActions.Clear();
            }
            for (int i = 0; i < currentActions.Count; i++)
                currentActions[i]();
            lock (delayedActions)
            {
                currentDelayedActions.Clear();
                for (int i = 0; i < delayedActions.Count; i++)
                {
                    if (delayedActions[i].timeMarker <= Time.time)
                        currentDelayedActions.Add(delayedActions[i]);
                }
                for (int i = 0; i < currentDelayedActions.Count; i++)
                    delayedActions.Remove(currentDelayedActions[i]);
            }
            for (int i = 0; i < currentDelayedActions.Count; i++)
                currentDelayedActions[i].delayedAction();

        }
    }
}
