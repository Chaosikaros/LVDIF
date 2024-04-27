using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class CameraMovement : MonoBehaviour
{
    public float moveSpeed = 0.2f;
    public float rotationSpeed = 0.3f;

    Vector3 anchorPoint;
    Quaternion anchorRot;
    private Vector3 initialPos;
    private Quaternion initialRotation;

    public Vector3 closePos;
    public Vector3 closeRotation;
    // Start is called before the first frame update
    void Start()
    {
        initialPos = transform.position;
        initialRotation = transform.rotation;
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 moveDirection = Vector3.zero;
        if (Input.GetKey(KeyCode.W))
            moveDirection += Vector3.forward * moveSpeed;
        if (Input.GetKey(KeyCode.S))
            moveDirection -= Vector3.forward * moveSpeed;
        if (Input.GetKey(KeyCode.D))
            moveDirection += Vector3.right * moveSpeed;
        if (Input.GetKey(KeyCode.A))
            moveDirection -= Vector3.right * moveSpeed;
        transform.Translate(moveDirection);

        if (Input.GetMouseButtonDown(1))
        {
            anchorPoint = new Vector3(Input.mousePosition.y, -Input.mousePosition.x);
            anchorRot = transform.rotation;
        }
        if (Input.GetMouseButton(1))
        {
            Quaternion anchorRotTemp = anchorRot;
            Vector3 dif = anchorPoint - new Vector3(Input.mousePosition.y, -Input.mousePosition.x);
            anchorRotTemp.eulerAngles += dif * rotationSpeed;
            transform.rotation = anchorRotTemp;
        }
        if (Input.GetKeyUp(KeyCode.Backspace))
        {
            transform.position = initialPos;
            transform.rotation = initialRotation;
        }
        if(Input.GetKeyUp(KeyCode.UpArrow))
        {
            transform.position = closePos;
            transform.rotation = Quaternion.Euler(closeRotation);
        }
    }
}
