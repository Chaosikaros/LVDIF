using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ChaosIkaros.LVDIF
{
    public class ChunkRayCast : MonoBehaviour
    {
        public int hitCounter = 0;
        public int hitPen = 0;
        public bool enableHapticForce = true;
        public Transform rayEnd;
        public Transform rayPenStart;
        public Transform rayPenEnd;
#if LVDIF_Haptic
        public HapticMaterial hM;
        public HapticPlugin hapticPlugin;
#endif
        // Start is called before the first frame update
        void Start()
        {

        }

        public void RayCastAll()
        {
            Ray ray = new Ray(transform.position, -(transform.position - rayEnd.position) * 1000);
            hitCounter = 0;
            RaycastHit[] hits = Physics.RaycastAll(ray);

            if (hits.Length > 0)
            {
                for (int i = 0; i < hits.Length; i++)
                {
                    if (hits[i].collider.CompareTag("Chunk"))
                        hitCounter++;
                }
            }

            Ray rayPen = new Ray(rayPenStart.position, rayPenEnd.position - rayPenStart.position);
            RaycastHit hit;
            hitPen = 0;
            if (Physics.Raycast(rayPen, out hit))
            {
                if (hit.collider.CompareTag("Pen"))
                    hitPen = 1;
                //else
                //    Debug.Log(hit.distance);
            }
            else
                hitPen = 1;

            if (hitPen == 0)
                Debug.DrawRay(rayPenStart.position, rayPenEnd.position - rayPenStart.position, Color.red);
            else
                Debug.DrawRay(rayPenStart.position, rayPenEnd.position - rayPenStart.position, Color.green);

            //Debug.Log("hc: " + hits.Length);

            if (hitPen == 0 && hitCounter == 0)
                Debug.DrawRay(transform.position, -(transform.position - rayEnd.position) * 1000, Color.green);
            else
                Debug.DrawRay(transform.position, -(transform.position - rayEnd.position) * 1000, Color.red);
        }

#if LVDIF_Haptic
    public void AddHapticForce()
    {
        int sFac, vFac, impCorrection;

        sFac = 1;
        vFac = 0;
        impCorrection = 1;

        HapticMaterial hapMat = hM;
        if (hapMat != null)
        {

            sFac = 1;
            vFac = 1;

            impCorrection = 1;
            HapticPlugin.ContactPointInfo contInfo = new HapticPlugin.ContactPointInfo();
            contInfo.Location = gameObject.transform.InverseTransformPoint(Vector3.one) / hapticPlugin.ScaleFactor;
            contInfo.Normal = gameObject.transform.InverseTransformVector(Vector3.one);
            contInfo.MaterialMass = hapMat.hMass;
            contInfo.MaterialStiffness = hapMat.hStiffness * sFac;
            contInfo.MaterialDamping = hapMat.hDamping * 0.0f;
            contInfo.MaterialFrictionStatic = hapMat.hFrictionS;
            contInfo.MaterialFrictionDynamic = hapMat.hFrictionD;
            contInfo.MaterialViscosity = hapMat.hViscosity * vFac;
            contInfo.MaterialSpring = hapMat.hSpringMag;
            contInfo.MaterialConstantForce = hapMat.hConstForceMag;
            contInfo.MatConstForceDir = hapMat.hConstForceDir;

            if (hapMat.UseContactNormalCF)
            {
                contInfo.MatConstForceDir = contInfo.Normal;
                if (hapMat.ContactNormalInverseCF)
                {
                    contInfo.MatConstForceDir *= -1.0f;
                }
            }

            contInfo.MaterialSpring = hapMat.hSpringMag;
            contInfo.MatSpringDir = hapMat.hSpringDir;
            contInfo.RigBodySpeed = 0.0f;
            contInfo.RigBodyVelocity = Vector3.zero;
            contInfo.RigBodyAngularVelocity = Vector3.zero;
            contInfo.RigBodyMass = 1.0f;
            contInfo.ColImpulse = Vector3.zero;
            contInfo.PhxDeltaTime = Time.fixedDeltaTime;
            contInfo.ImpulseDepth = 0.0f;
            contInfo.ColliderName = gameObject.name;
            if (!hapticPlugin.isInCPList(contInfo))
            {
                hapticPlugin.ContactPointsInfo.Add(contInfo);
            }
            //Debug.DrawRay(contInfo.Location*ScaleFactor, contInfo.Normal/10, Color.green, 2, false);
        }
    }
#endif
        private void FixedUpdate()
        {
            RayCastAll();
#if LVDIF_Haptic
        if (enableHapticForce && hitPen == 0 && hitCounter == 0)
            AddHapticForce();
#endif
        }

        // Update is called once per frame
        void Update()
        {

        }
    }
}
