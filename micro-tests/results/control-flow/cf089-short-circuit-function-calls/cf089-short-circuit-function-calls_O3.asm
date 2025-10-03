
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf089-short-circuit-function-calls/cf089-short-circuit-function-calls_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z5func1v>:
100000360: 52800000    	mov	w0, #0x0                ; =0
100000364: d65f03c0    	ret

0000000100000368 <__Z5func2v>:
100000368: 52800020    	mov	w0, #0x1                ; =1
10000036c: d65f03c0    	ret

0000000100000370 <__Z5func3v>:
100000370: 52800040    	mov	w0, #0x2                ; =2
100000374: d65f03c0    	ret

0000000100000378 <__Z27test_function_short_circuitv>:
100000378: 52800020    	mov	w0, #0x1                ; =1
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: 52800020    	mov	w0, #0x1                ; =1
100000384: d65f03c0    	ret
