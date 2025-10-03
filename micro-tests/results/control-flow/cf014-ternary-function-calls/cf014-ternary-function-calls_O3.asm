
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf014-ternary-function-calls/cf014-ternary-function-calls_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z13positive_funcv>:
100000360: 52800020    	mov	w0, #0x1                ; =1
100000364: d65f03c0    	ret

0000000100000368 <__Z13negative_funcv>:
100000368: 12800000    	mov	w0, #-0x1               ; =-1
10000036c: d65f03c0    	ret

0000000100000370 <__Z18test_ternary_callsi>:
100000370: 7100041f    	cmp	w0, #0x1
100000374: 12800008    	mov	w8, #-0x1               ; =-1
100000378: 5a88b500    	cneg	w0, w8, ge
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: 52800020    	mov	w0, #0x1                ; =1
100000384: d65f03c0    	ret
