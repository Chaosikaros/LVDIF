using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ChaosIkaros.LVDIF
{
    public class InputVolume : MonoBehaviour
    {
        public Transform vertexA;
        public Transform vertexB;
        // Start is called before the first frame update
        void Start()
        {

        }

        private void OnDrawGizmos()
        {
            Gizmos.color = new Color(0, 1, 1, 0.2f);
            Gizmos.DrawCube((vertexA.position + vertexB.position) / 2,
                Vector3.one * Vector3.Distance(vertexA.position, vertexB.position) / 1.73205f);

            Gizmos.color = new Color(0, 1, 0, 0.5f);
            Gizmos.DrawCube(vertexA.position,
                Vector3.one * Vector3.Distance(vertexA.position, vertexB.position) / 30f);
            Gizmos.color = new Color(0, 1, 0, 0.5f);
            Gizmos.DrawCube(vertexB.position,
                Vector3.one * Vector3.Distance(vertexA.position, vertexB.position) / 30f);
        }

        public Vector3 WorldToLocal(Vector3 input)
        {
            Vector3 inputT = transform.InverseTransformPoint(input);
            return new Vector3(Mathf.Clamp(inputT.x, vertexA.localPosition.x, vertexB.localPosition.x),
                Mathf.Clamp(inputT.y, vertexA.localPosition.y, vertexB.localPosition.y),
                Mathf.Clamp(inputT.z, vertexA.localPosition.z, vertexB.localPosition.z));
        }

        public Vector3 LocalToWorld(Vector3 input)
        {
            Vector3 inputT = transform.TransformPoint(input);
            return new Vector3(Mathf.Clamp(inputT.x, vertexA.position.x, vertexB.position.x),
                Mathf.Clamp(inputT.y, vertexA.position.y, vertexB.position.y),
                Mathf.Clamp(inputT.z, vertexA.position.z, vertexB.position.z));
        }

        public void ClampVertex()
        {
            Vector3 inputA = vertexA.localPosition;
            Vector3 inputB = vertexB.localPosition;
            if (inputB.x < inputA.x)
                inputB.x = inputA.x;
            if (inputB.y < inputA.y)
                inputB.y = inputA.y;
            if (inputB.z < inputA.z)
                inputB.z = inputA.z;
            vertexB.localPosition = inputB;
        }

        private void OnValidate()
        {
            ClampVertex();
        }

        // Update is called once per frame
        void Update()
        {

        }
    }
}
