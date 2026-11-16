(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)


(* Seven bodies of code in which Simplify unboxed a variant that is the
   parameter of a join, and the validation of each once
   [unbox_variant_params] is in the round of the term before.

   This file is to be reviewed: the bodies, against the sources below;
   the round; and the statements.

   [option_heap] is

     external on_heap : 'a -> 'a = "%identity"

     let[@inline never] f : bool -> int -> int =
      fun c x ->
       let v = if c then Some (x + 1) else None in
       match on_heap v with None -> 0 | Some y -> y + 2

   One branch makes the block of [Some] and the other gives the
   constant [None]; the continuation of the [let] tests whether its
   parameter is an immediate and, where it is not, loads the field.
   Simplify gives that continuation two parameters and not the option:
   whether it is an immediate, and the field. The branch of [Some]
   gives 0 and what it computed, and makes no block; the branch of
   [None] gives 1 and the constant [poison] of the kind of values.
   ([on_heap] is there so that the option is not allocated in a
   region: the model has no regions yet. The front end removes it.)

   [list_heap] is the same of a list, [[x + 1]] or [[]], matched by
   [[] -> x + 3 | y :: _ -> y + 2]: both arms of the match are
   continuations, and only the head of the list is read.

   [shape_heap] is

     type shape = Empty | Full | Pair of int * int

     let[@inline never] f : bool -> int -> int =
      fun c x ->
       let v = if c then Pair (x + 1, x) else Full in
       match on_heap v with Empty -> 0 | Full -> 1 | Pair (a, b) -> a + b

   The front end returns the value itself for the two constant
   constructors, so the parameter stays, and the block is still made;
   the continuation gets three more parameters, whether the value is
   an immediate and the two fields, and the branch of [Full] gives
   [poison] for both fields.

   [shape2_heap] is [shape_heap] with [Empty -> 10 | Full -> x + 20]
   for the two constant constructors. The handler then reads which
   constant constructor its parameter is, by an untagging, under the
   test that it is an immediate; Simplify makes that a parameter too,
   the first, and the branch of [Pair] gives 0 for it. The value is
   no longer a parameter and no block is made.

   [nested_heap] is an option of a pair, [Some (x + 1, x)] or [None],
   matched by [None -> 0 | Some (a, b) -> a + b]. Simplify unboxes the
   option and then the pair: the continuation gets whether the value
   is an immediate and the two fields of the pair, and the branch of
   [None] gives [poison] for both. On the term before, the pair is
   first made a parameter, which the branch of [None] gives [poison];
   its two fields are then read from that parameter, and a load has no
   value on the constant [poison] either.

   [either_heap] is

     type either = Left of int | Right of int * int

     let[@inline never] f : bool -> int -> int =
      fun c x ->
       let v = if c then Left (x + 1) else Right (x, x + 2) in
       match on_heap v with Left y -> y + 3 | Right (y, z) -> y + z

   Both branches make a block, of two tags, and the continuation of
   the [let] goes by the tag to the continuation of one constructor,
   which loads its fields. Simplify gives the continuation the tag and
   the fields of both constructors, those of [Right] first, and each
   branch gives [poison] for the fields of the other constructor. The
   loads say nothing of the tag, so the load of the field of [Left]
   has a value on the block of [Right]: here the pass gives [poison]
   because the branch of [Right] gives the tag 1, with which the
   handler does not come to the continuation of [Left]
   (FlambdaUnboxVariantsSpec.never_reads).

   [int64_option_heap] is

     let[@inline never] f : bool -> int64 -> int64 =
      fun c x ->
       let v = if c then Some (add x 1L) else None in
       match on_heap v with None -> 0L | Some y -> add y 2L

   with the arithmetic of %int64_add. The branch of [Some] boxes a
   number and makes the block of it; the continuation of the [let]
   loads the field and unboxes it. Simplify unboxes the option and then
   the number: the continuation gets whether the value is an immediate
   and the number, the branch of [Some] gives the number that it
   computed, and boxes nothing, and the branch of [None] gives [poison]
   of the kind of the number. On the term before, the field is first
   made a parameter, and then its unboxing, which is under the test:
   the branch of [Some] gives the number in the box that it made
   (FlambdaUnboxVariantsSpec.made_number). The two constants [1L] and
   [2L] are symbols that the unit binds to boxed numbers, which the
   body unboxes; they are written here as the numbers, and the boxed
   [0L] that the branch of [None] returns as the symbol 1.

   [while_two_tests] is

     let result =
       let n = opaque 10 in
       let i = ref 0 in
       let s = ref 0 in
       while !i < n && !s < 20 do
         s := !s + !i;
         i := !i + 1
       done;
       !s

   The [&&] is a join in the handler of the loop: the continuation
   after the two tests takes a boolean, which the first test gives as
   the constant [false] and the second as its comparison, tagged; it
   untags the boolean for its [Switch]. Simplify gives that
   continuation the number untagged: 0 from the first test, the
   comparison itself from the second, and nothing is tagged. On the
   term before, the untagging is not the first thing in the handler
   of the join, as the continuation of the body of the loop is bound
   above it; so the argument must be certain to be a naked immediate:
   the constant, and the number that the second test tagged
   (FlambdaUnboxVariantsSpec.made_number). The body here is the
   handler of the loop. The front end has made the two references
   parameters of the loop, 1 and 2, and of each continuation in it;
   the bound [n] is the variable 3, the loop is the continuation 2
   and its exit the continuation 1, which is given the result alone.

   [shape_constant] is the body of a unit in which

     type shape = Empty | Full | Box of int

     let v = if c then Box 8 else Empty in
     match v with Empty -> 1 | Full -> 2 | Box y -> y

   is not in a function. The block [Box 8] is a constant, so the unit
   binds a symbol to it, and the join is given that symbol or the
   constant [Empty]. Simplify gives the continuation whether the value
   is an immediate and the field, 8 from the symbol and [poison] from
   [Empty]; the constant constructor it finds to be the first on both
   paths. On the term before, the untagging of the value is under the
   test of whether it is an immediate, and the path of the symbol has
   nothing to untag: a symbol is never an immediate
   (FlambdaUnboxVariantsSpec.no_value_on), so that path gives 0, as
   the other does, and the parameter is then bound to it and the
   [Switch] on it folded. The load is given the field of the symbol,
   a constant, by what is known of the static block in scope. The
   condition is the variable 1 and the return continuation is 1; the
   symbol is 1, and its binding, which nothing uses in the end, is
   kept in both bodies.

   The bodies are those of the function of each source: the body
   before Simplify and the body after. They are written by hand from
   the two terms that the compiler prints ([-drawfexpr-to] and
   [-dfexpr-to]). The binding of the depth variable of the code, which
   has nothing in the model, is left out; the names are numbered here.
   The units themselves are in
   middle_end/flambda2/tests/semantics/ocaml/ ([option_heap.ml] and the
   others), where the validation is run on the whole of each.

   [round_with_variants] is
   FlambdaUnboxParamsExamplesSpec.round_with_unboxing with
   [unbox_variant_params] for [unbox_known_params]. It is here for the
   examples only; the validation does not run it.

   What is said of each body: the body before, after five such rounds,
   is the body after up to bound names ([..._validated_statement]);
   after five rounds of FlambdaValidateSpec.normalise_expr_before,
   which has no unboxing of parameters, it is not
   ([..._not_validated_today_statement]: the statements were written
   before the validation ran the pass, and their names are kept); and
   the body after refines
   the body before, under all environments ([..._refines_statement]),
   which follows from the first by the theorems of the passes and of
   the comparison. So the constant [poison] that Simplify gives where
   there is no field is where the pass gives it. Of [poison] that says
   what the refinement says and no more: a run of the body before that
   is defined is matched by the body after; where the body before is
   undefined, as it is if it comes to a read that has no value,
   nothing is claimed of the body after.

   The two bodies that FlambdaUnboxParamsExamplesSpec.v validates with
   its round are validated with this one too
   ([earlier_examples_validated_statement]).

   The validation runs the pass in the first phase of its
   normalisation of the term before
   (FlambdaValidateSpec.normalise_expr_params). Five rounds of
   normalise_expr_params take each of the seven bodies of this file to
   the body after, and the three of the other file too, [pair_built]
   among them, which the round of that file does not validate
   ([bodies_validated_by_first_phase_statement]). They take the body
   of [while_two_tests] to its body after as well
   ([while_two_tests_validated_statement]), so that body after refines
   the body before ([while_two_tests_refines_statement]); the round
   of this file does not, as it binds no parameter that every
   application gives the same thing. The same is said of
   [shape_constant] ([shape_constant_validated_statement],
   [shape_constant_refines_statement]). In the validation each such
   round for a whole term is followed by a step that these statements
   do not have (FlambdaValidateSpec.first_phase_round): it makes the
   applications of a join go to the arms of its Switch, which the
   body that Simplify gave keeps.

   The last examples are whole terms, with a symbol that a function,
   or a loop, is given where it is bound. [constant_join] is the body
   of the function of

     let[@inline never] f : bool -> int -> int =
      fun c x ->
       let v = if c then Some 8 else None in
       match v with None -> x | Some y -> x + y

   The block [Some 8] is a constant, which closure conversion makes a
   symbol of the unit, bound before the code of [f]; the join in the
   body is given that symbol or the constant [None]. Simplify gives
   the join whether the value is an immediate and the field: 8 from
   the symbol and [poison] from [None]. The body alone says nothing of
   the symbol. A round of the first phase for a whole term
   (FlambdaValidateSpec.normalise_params_round) gives the round for
   the body of code what the static bindings in scope make known
   (FlambdaUnboxInScopeSpec.rewrite_code_in_scope), and the round for
   the handler of a loop likewise
   (FlambdaUnboxInScopeSpec.rewrite_rec_handlers_in_scope). So:
   - [constant_in_code] binds the symbol and the code in one group.
     Five such rounds take it to the term with the join unboxed
     ([constant_in_code_validated_statement]), which therefore refines
     it ([constant_in_code_refines_statement]);
   - [constant_below_code] binds the code first and the symbol under
     it. The symbol is not in scope where the code is bound, and five
     rounds leave the term as it is
     ([constant_below_code_left_statement]);
   - [constant_in_loop] has the same join in the handler of a
     recursive continuation under the binding of the symbol; the two
     parameters of the handler are 1 and 2, and it is applied once, to
     the variables 1 and 2 of the term. Five rounds unbox the join
     there ([constant_in_loop_validated_statement],
     [constant_in_loop_refines_statement]).
   In the three the symbol is 1 and the code id is 1; the parameters
   of the code are [c], 1, and [x], 2, its closure variable is 3 and
   its return and exception continuations are 1 and 2. In the two
   terms with code, the code is not called: the body of the term
   returns a constant to the continuation 1, and only the bindings
   matter. The term with the loop applies its handler, which returns
   [x], or [x] and 8 added, to the continuation 1.

   Three bodies more, for what the pass gives where Simplify's rule
   is by the box and not by the path that the handler takes:
   - [pair_constant] is the body of

       fun c x ->
        match if c then x, true else 0, false with
        | v, true -> v
        | _, false -> -7

     The handler of the join reads the first field only where the
     second is [true], and the arm of the constant pair gives
     [false]. Simplify gives that arm the first field of its pair, 0:
     a pair has one tag, and its fields are given wherever the pair
     is. The second field is no test of the pair, so the pass does
     not take it to show that the arm does not come to the read
     ([pair_constant_validated_statement],
     [pair_constant_refines_statement]).
   - [option_pair] is the body of

       fun c x ->
        match if c then Some (x, true) else None with
        | Some (n, b) -> if b then n else 0
        | None -> -1

     The arm of [None] gives [poison] for the untagging of the
     boolean, as for the fields: the boolean is a number, not the
     constant constructor of a variant, of which nothing tests
     [Is_int] ([option_pair_validated_statement],
     [option_pair_refines_statement]).
   - [option_fields] is the body of

       fun c x ->
        match if c then Some (3, x) else None with
        | Some (low, high) -> high + low
        | None -> 0

     The handler loads the first field of the pair, then the second;
     the new parameters are the second, then the first, as Simplify
     has them ([option_fields_validated_statement],
     [option_fields_refines_statement]).
   In the three the continuation of the join is 2 and its parameter
   is 3.

   One body more, for a load that the pass leaves. [two_tags] is the
   body of

       type side = Left of int | Right of int | Neither

       fun c x ->
        match if c then Left x else Right x with
        | Left v | Right v -> v + 1
        | Neither -> 0

   Both arms make a block, of two tags, and the handler of the join
   tests whether its parameter is an immediate and then loads the
   field that the two constructors share, without going by the tag.
   Simplify finds that the value is an immediate on neither arm and
   removes the test; the two blocks, the parameter and the load
   stay. The pass makes the test a parameter, which both arms give 0
   and the round then removes, and leaves the load: blocks of two
   tags come to it (FlambdaUnboxVariantsSpec.loads_of_two_tags). So
   the rounds take the body before to the body after
   ([two_tags_validated_statement], [two_tags_refines_statement]).
   The continuation of the join is 2 and its parameter is 3; the
   continuations that make the blocks are 4 and 5. The front end
   allocates the two blocks in a region that it opens in the body and
   closes after the join; the model has no regions, and the bodies
   are written without it.

   One body more, for an argument that the term has already read.
   [int64_max] is the body of

       fun a b ->
        let m = if less a b then b else a in
        add m 1L

   on boxed 64-bit integers. The comparison unboxes [a] and [b]; each
   arm gives the join one of the two boxes, and the handler unboxes
   what it is given. Simplify gives the join the number: on each arm
   the one that the comparison unboxed, which is in scope. Neither box
   is made in the body, so nothing is known of what is in it but that
   a [Let] above has read it (FlambdaUnboxParamsSpec.reads_under_let),
   and the pass gives the variable of that [Let]
   ([int64_max_validated_statement], [int64_max_refines_statement]).
   The parameters of the code are [a], 1, and [b], 2; the continuation
   of the join is 2 and its parameter is 3; the constant [1L] is
   written as the number, as the constants of [int64_option_heap] are.

   One body more, for what a handler reads in two places.
   [matched_twice] is the body of

       fun c x ->
        let o = if c then Some (x + 1) else None in
        let a = match o with None -> 0 | Some y -> y in
        let b = match o with None -> 10 | Some y -> y * 2 in
        a + b

   The handler of the join tests its parameter twice and loads its
   field twice, the second test and the second load in the
   continuation that the first match goes to. Simplify gives the join
   two parameters, whether the option is an immediate and its
   contents, and each test and each load is of them. The pass makes
   the first test a parameter and the second a [Let] of it, and the
   same for the loads (FlambdaUnboxVariantsSpec.share_reads); the
   rounds follow those [Let]s
   ([matched_twice_validated_statement],
   [matched_twice_refines_statement]). The parameters of the code are
   [c], 1, and [x], 2; the continuation of the join is 2 and its
   parameter is 3; the continuations of the two matches are 3 and 4,
   with parameters 4 and 5. The region of the block is left out, as
   in [two_tags].

   One body more, for a loop. [pair_loop] is the body of

       fun n ->
        let rec turn i p =
          let a, b = p in
          if i + 1 < n then turn (i + 1) (b, a + b) else b
        in
        turn 1 (0, 1)

   written by hand in the shape of what the front end makes of a
   reference that holds a pair and that a [for] loop sets at every
   turn: the pair is a parameter of the loop, which is entered with a
   pair that a [Let] makes and given, at each turn, a pair that a
   [Let] of the handler makes. Simplify gives the loop the two fields
   for parameters, the second first. The pass that the round applies
   after the unboxing of joins
   (FlambdaUnboxLoopsSpec.unbox_loops_knowing) makes the two loads,
   which the handler begins with, parameters, and the rewrites that
   come after it remove the pairs and the parameter that held them
   ([pair_loop_validated_statement], [pair_loop_refines_statement]).
   The parameter of the code is [n], 1; the continuation of the loop
   is 2 and its parameters are the counter, 3, and the pair, 4.

   One term more, for a box that one join passes on to another.
   [int64_passed_on] is the code of

       fun a b x ->
        let t = if b then add x 1L else add x 2L in
        let y = if a then t else 3L in
        add y y

   on boxed 64-bit integers, under the binding of the symbol of the
   constant [3L]. The join of the first [if] is given a box that each
   arm makes, and only gives it to the join of the second, whose
   other arm gives the symbol; the handler of the second unboxes what
   it is given. Simplify gives both joins the number: the first, what
   each arm computed; the second, the parameter of the first or the
   constant 3. The handler of the first join does not read its
   parameter, and an arm of the second gives a parameter, of which
   the term shows nothing: the unboxing of the parameters of joins
   leaves both. The pass that the round applies after the untagging
   (FlambdaUnboxTogetherSpec.unbox_together_knowing) unboxes the two
   parameters at once: an argument is then a parameter that is
   unboxed with them, the number that a [Let] in scope boxed, or the
   number of the symbol ([int64_passed_on_validated_statement],
   [int64_passed_on_refines_statement]; the rounds are those for a
   whole term, in which the code knows the symbol). The parameters of
   the code are [a], 1, [b], 2, and [x], 3; its closure variable is 4
   and its return and exception continuations are 1 and 2; the joins
   are the continuations 3 and 4, with parameters 5 and 6; the symbol
   is 1 and the code id is 1. The constants [1L] and [2L] are written
   as the numbers, as those of [int64_option_heap] are. The code is
   not called: the body of the term returns a constant, and only the
   bindings matter.

   The proofs are in FlambdaUnboxVariantsExamplesProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaSemanticsSpec FlambdaCompareSpec
  FlambdaNormaliseSpec FlambdaBlockLoadSpec FlambdaLoopAliasesSpec
  FlambdaShortcutSpec FlambdaSinkContsSpec FlambdaUnboxSpec
  FlambdaUnusedContsSpec FlambdaUnusedParamsSpec FlambdaValidateSpec
  FlambdaUnboxParamsSpec FlambdaUnboxParamsExamplesSpec
  FlambdaUnboxVariantsSpec FlambdaUnboxInScopeSpec.
From OxCamlFlambda2Types Require Import FlambdaAliasesSpec.
Import ListNotations.
Open Scope positive_scope.

(* ======================================================================== *)
(* The round                                                                *)
(* ======================================================================== *)

Definition round_with_variants (e : expr) : expr :=
  let e :=
    shortcut_switches
      (thread_switches
         (inline_tail_conts
            (sink_let_conts
               (remove_unused_conts (collapse_switches (fold_constants e))))))
  in
  let e :=
    unbox_variant_params
      (follow_aliases_in_loops []
         (follow_aliases [] (unbox_known_numbers (load_known_blocks e))))
  in
  drop_unused_params
    (remove_unused_defined_lets [] (remove_unused_lets (fold_constants e))).

(* ======================================================================== *)
(* What the bodies are made of                                              *)
(* ======================================================================== *)

(* The test that the match of a variant begins with. *)
Definition is_int (s : simple) : named := Prim (Unary (Is_int true)) [s].

Definition tag_immediate (s : simple) : named :=
  Prim (Unary Tag_immediate) [s].

Definition block_of_one (contents : simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values 0))) [contents].

Definition naked (i : Z) : simple := Const (Naked_immediate i).

Definition poison_value : simple := Const (Poison Value).

(* In each body the parameters of the code are [c], 1, and [x], 2, and
   its return continuation is 1. The continuation of the join is 2,
   its parameter is 3, and the continuations of the branch that makes
   a block are 3 and 4. *)

(* ======================================================================== *)
(* An option                                                                *)
(* ======================================================================== *)

Definition option_heap_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 4 Value (field 0 3)
                (Let 5 Value (add_tagged (Var 4) (tagged 2))
                   (Apply_cont (jump 1 [Var 5])))))
          (Let 6 naked_immediate (is_int (Var 3))
             (Let 7 Value (tag_immediate (Var 6))
                (Let_cont_non_recursive 6
                   (Handler [] (make_handler_metadata [])
                      (Apply_cont (jump 5 [])))
                   (Let 8 naked_immediate (untag (Var 7))
                      (Switch (Var 8)
                         [(0%Z, jump 6 []); (1%Z, jump 1 [tagged 0])])))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 9 Value (add_tagged (Var 2) (tagged 1))
             (Let 10 Value (block_of_one (Var 9))
                (Apply_cont (jump 2 [Var 10])))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata []) (Apply_cont (jump 3 [])))
          (Let 11 naked_immediate (untag (Var 1))
             (Switch (Var 11) [(0%Z, jump 2 [tagged 0]); (1%Z, jump 4 [])])))).

Definition option_heap_after : expr :=
  Let_cont_non_recursive 2
    (Handler [6; 4] (make_handler_metadata [naked_immediate; Value])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 5 Value (add_tagged (Var 4) (tagged 2))
                (Apply_cont (jump 1 [Var 5]))))
          (Switch (Var 6) [(0%Z, jump 5 []); (1%Z, jump 1 [tagged 0])])))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 9 Value (add_tagged (Var 2) (tagged 1))
             (Apply_cont (jump 2 [naked 0; Var 9]))))
       (Let 11 naked_immediate (untag (Var 1))
          (Switch (Var 11)
             [(0%Z, jump 2 [naked 1; poison_value]); (1%Z, jump 3 [])]))).

Definition option_heap_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_variants option_heap_before)
    option_heap_after
  = true.

Definition option_heap_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before option_heap_before)
    option_heap_after
  = false.

Definition option_heap_refines_statement : Prop :=
  expr_refines option_heap_before option_heap_after.

(* ======================================================================== *)
(* A list                                                                   *)
(* ======================================================================== *)

Definition list_heap_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 9 Value (add_tagged (Var 2) (tagged 3))
                (Apply_cont (jump 1 [Var 9]))))
          (Let_cont_non_recursive 6
             (Handler [] (make_handler_metadata [])
                (Let 7 Value (field 0 3)
                   (Let 8 Value (add_tagged (Var 7) (tagged 2))
                      (Apply_cont (jump 1 [Var 8])))))
             (Let 4 naked_immediate (is_int (Var 3))
                (Let 5 Value (tag_immediate (Var 4))
                   (Let_cont_non_recursive 7
                      (Handler [] (make_handler_metadata [])
                         (Apply_cont (jump 6 [])))
                      (Let_cont_non_recursive 8
                         (Handler [] (make_handler_metadata [])
                            (Apply_cont (jump 5 [])))
                         (Let 6 naked_immediate (untag (Var 5))
                            (Switch (Var 6)
                               [(0%Z, jump 7 []); (1%Z, jump 8 [])])))))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 10 Value (add_tagged (Var 2) (tagged 1))
             (Let 11 Value (pair (Var 10) (tagged 0))
                (Apply_cont (jump 2 [Var 11])))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata []) (Apply_cont (jump 3 [])))
          (Let 12 naked_immediate (untag (Var 1))
             (Switch (Var 12) [(0%Z, jump 2 [tagged 0]); (1%Z, jump 4 [])])))).

Definition list_heap_after : expr :=
  Let_cont_non_recursive 2
    (Handler [4; 7] (make_handler_metadata [naked_immediate; Value])
       (Let_cont_non_recursive 6
          (Handler [] (make_handler_metadata [])
             (Let 8 Value (add_tagged (Var 7) (tagged 2))
                (Apply_cont (jump 1 [Var 8]))))
          (Let_cont_non_recursive 5
             (Handler [] (make_handler_metadata [])
                (Let 9 Value (add_tagged (Var 2) (tagged 3))
                   (Apply_cont (jump 1 [Var 9]))))
             (Switch (Var 4) [(0%Z, jump 6 []); (1%Z, jump 5 [])]))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 10 Value (add_tagged (Var 2) (tagged 1))
             (Apply_cont (jump 2 [naked 0; Var 10]))))
       (Let 12 naked_immediate (untag (Var 1))
          (Switch (Var 12)
             [(0%Z, jump 2 [naked 1; poison_value]); (1%Z, jump 3 [])]))).

Definition list_heap_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_variants list_heap_before)
    list_heap_after
  = true.

Definition list_heap_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before list_heap_before)
    list_heap_after
  = false.

Definition list_heap_refines_statement : Prop :=
  expr_refines list_heap_before list_heap_after.

(* ======================================================================== *)
(* Two constant constructors and a pair                                     *)
(* ======================================================================== *)

Definition shape_heap_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Apply_cont (jump 1 [Var 3])))
          (Let_cont_non_recursive 6
             (Handler [] (make_handler_metadata [])
                (Let 7 Value (field 1 3)
                   (Let 8 Value (field 0 3)
                      (Let 9 Value (add_tagged (Var 8) (Var 7))
                         (Apply_cont (jump 1 [Var 9]))))))
             (Let 4 naked_immediate (is_int (Var 3))
                (Let 5 Value (tag_immediate (Var 4))
                   (Let_cont_non_recursive 7
                      (Handler [] (make_handler_metadata [])
                         (Apply_cont (jump 6 [])))
                      (Let_cont_non_recursive 8
                         (Handler [] (make_handler_metadata [])
                            (Apply_cont (jump 5 [])))
                         (Let 6 naked_immediate (untag (Var 5))
                            (Switch (Var 6)
                               [(0%Z, jump 7 []); (1%Z, jump 8 [])])))))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 10 Value (add_tagged (Var 2) (tagged 1))
             (Let 11 Value (pair (Var 10) (Var 2))
                (Apply_cont (jump 2 [Var 11])))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata []) (Apply_cont (jump 3 [])))
          (Let 12 naked_immediate (untag (Var 1))
             (Switch (Var 12) [(0%Z, jump 2 [tagged 1]); (1%Z, jump 4 [])])))).

Definition shape_heap_after : expr :=
  Let_cont_non_recursive 2
    (Handler [3; 4; 7; 8]
       (make_handler_metadata [Value; naked_immediate; Value; Value])
       (Let_cont_non_recursive 6
          (Handler [] (make_handler_metadata [])
             (Let 9 Value (add_tagged (Var 8) (Var 7))
                (Apply_cont (jump 1 [Var 9]))))
          (Switch (Var 4) [(0%Z, jump 6 []); (1%Z, jump 1 [Var 3])])))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 10 Value (add_tagged (Var 2) (tagged 1))
             (Let 11 Value (pair (Var 10) (Var 2))
                (Apply_cont (jump 2 [Var 11; naked 0; Var 2; Var 10])))))
       (Let 12 naked_immediate (untag (Var 1))
          (Switch (Var 12)
             [(0%Z, jump 2 [tagged 1; naked 1; poison_value; poison_value]);
              (1%Z, jump 3 [])]))).

Definition shape_heap_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_variants shape_heap_before)
    shape_heap_after
  = true.

Definition shape_heap_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before shape_heap_before)
    shape_heap_after
  = false.

Definition shape_heap_refines_statement : Prop :=
  expr_refines shape_heap_before shape_heap_after.

(* ======================================================================== *)
(* Two constant constructors that are told apart, and a pair                *)
(* ======================================================================== *)

Definition shape2_heap_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let_cont_non_recursive 9
                (Handler [] (make_handler_metadata [])
                   (Let 11 Value (add_tagged (Var 2) (tagged 20))
                      (Apply_cont (jump 1 [Var 11]))))
                (Let_cont_non_recursive 10
                   (Handler [] (make_handler_metadata [])
                      (Apply_cont (jump 9 [])))
                   (Let 10 naked_immediate (untag (Var 3))
                      (Switch (Var 10)
                         [(0%Z, jump 1 [tagged 10]); (1%Z, jump 10 [])])))))
          (Let_cont_non_recursive 6
             (Handler [] (make_handler_metadata [])
                (Let 7 Value (field 1 3)
                   (Let 8 Value (field 0 3)
                      (Let 9 Value (add_tagged (Var 8) (Var 7))
                         (Apply_cont (jump 1 [Var 9]))))))
             (Let 4 naked_immediate (is_int (Var 3))
                (Let 5 Value (tag_immediate (Var 4))
                   (Let_cont_non_recursive 7
                      (Handler [] (make_handler_metadata [])
                         (Apply_cont (jump 6 [])))
                      (Let_cont_non_recursive 8
                         (Handler [] (make_handler_metadata [])
                            (Apply_cont (jump 5 [])))
                         (Let 6 naked_immediate (untag (Var 5))
                            (Switch (Var 6)
                               [(0%Z, jump 7 []); (1%Z, jump 8 [])])))))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 12 Value (add_tagged (Var 2) (tagged 1))
             (Let 13 Value (pair (Var 12) (Var 2))
                (Apply_cont (jump 2 [Var 13])))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata []) (Apply_cont (jump 3 [])))
          (Let 14 naked_immediate (untag (Var 1))
             (Switch (Var 14) [(0%Z, jump 2 [tagged 1]); (1%Z, jump 4 [])])))).

Definition shape2_heap_after : expr :=
  Let_cont_non_recursive 2
    (Handler [10; 4; 7; 8]
       (make_handler_metadata
          [naked_immediate; naked_immediate; Value; Value])
       (Let_cont_non_recursive 6
          (Handler [] (make_handler_metadata [])
             (Let 9 Value (add_tagged (Var 8) (Var 7))
                (Apply_cont (jump 1 [Var 9]))))
          (Let_cont_non_recursive 5
             (Handler [] (make_handler_metadata [])
                (Let_cont_non_recursive 9
                   (Handler [] (make_handler_metadata [])
                      (Let 11 Value (add_tagged (Var 2) (tagged 20))
                         (Apply_cont (jump 1 [Var 11]))))
                   (Switch (Var 10)
                      [(0%Z, jump 1 [tagged 10]); (1%Z, jump 9 [])])))
             (Switch (Var 4) [(0%Z, jump 6 []); (1%Z, jump 5 [])]))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 12 Value (add_tagged (Var 2) (tagged 1))
             (Apply_cont (jump 2 [naked 0; naked 0; Var 2; Var 12]))))
       (Let 14 naked_immediate (untag (Var 1))
          (Switch (Var 14)
             [(0%Z, jump 2 [naked 1; naked 1; poison_value; poison_value]);
              (1%Z, jump 3 [])]))).

Definition shape2_heap_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_variants shape2_heap_before)
    shape2_heap_after
  = true.

Definition shape2_heap_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before shape2_heap_before)
    shape2_heap_after
  = false.

Definition shape2_heap_refines_statement : Prop :=
  expr_refines shape2_heap_before shape2_heap_after.

(* ======================================================================== *)
(* An option of a pair                                                      *)
(* ======================================================================== *)

Definition nested_heap_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 7 Value (field 0 3)
                (Let 8 Value (field 1 7)
                   (Let 9 Value (field 0 7)
                      (Let 10 Value (add_tagged (Var 9) (Var 8))
                         (Apply_cont (jump 1 [Var 10])))))))
          (Let 4 naked_immediate (is_int (Var 3))
             (Let 5 Value (tag_immediate (Var 4))
                (Let_cont_non_recursive 6
                   (Handler [] (make_handler_metadata [])
                      (Apply_cont (jump 5 [])))
                   (Let 6 naked_immediate (untag (Var 5))
                      (Switch (Var 6)
                         [(0%Z, jump 6 []); (1%Z, jump 1 [tagged 0])])))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 11 Value (add_tagged (Var 2) (tagged 1))
             (Let 12 Value (pair (Var 11) (Var 2))
                (Let 13 Value (block_of_one (Var 12))
                   (Apply_cont (jump 2 [Var 13]))))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata []) (Apply_cont (jump 3 [])))
          (Let 14 naked_immediate (untag (Var 1))
             (Switch (Var 14) [(0%Z, jump 2 [tagged 0]); (1%Z, jump 4 [])])))).

Definition nested_heap_after : expr :=
  Let_cont_non_recursive 2
    (Handler [4; 8; 9] (make_handler_metadata [naked_immediate; Value; Value])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 10 Value (add_tagged (Var 9) (Var 8))
                (Apply_cont (jump 1 [Var 10]))))
          (Switch (Var 4) [(0%Z, jump 5 []); (1%Z, jump 1 [tagged 0])])))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 11 Value (add_tagged (Var 2) (tagged 1))
             (Apply_cont (jump 2 [naked 0; Var 2; Var 11]))))
       (Let 14 naked_immediate (untag (Var 1))
          (Switch (Var 14)
             [(0%Z, jump 2 [naked 1; poison_value; poison_value]);
              (1%Z, jump 3 [])]))).

Definition nested_heap_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_variants nested_heap_before)
    nested_heap_after
  = true.

Definition nested_heap_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before nested_heap_before)
    nested_heap_after
  = false.

Definition nested_heap_refines_statement : Prop :=
  expr_refines nested_heap_before nested_heap_after.

(* ======================================================================== *)
(* Two constructors that hold fields                                        *)
(* ======================================================================== *)

Definition get_tag (s : simple) : named := Prim (Unary Get_tag) [s].

Definition block_of_two (tag : Z) (first second : simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values tag))) [first; second].

Definition either_heap_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 7
          (Handler [] (make_handler_metadata [])
             (Let 10 Value (field 0 3)
                (Let 11 Value (add_tagged (Var 10) (tagged 3))
                   (Apply_cont (jump 1 [Var 11])))))
          (Let_cont_non_recursive 8
             (Handler [] (make_handler_metadata [])
                (Let 7 Value (field 1 3)
                   (Let 8 Value (field 0 3)
                      (Let 9 Value (add_tagged (Var 8) (Var 7))
                         (Apply_cont (jump 1 [Var 9]))))))
             (Let 4 naked_immediate (get_tag (Var 3))
                (Let 5 Value (tag_immediate (Var 4))
                   (Let 6 naked_immediate (untag (Var 5))
                      (Switch (Var 6)
                         [(0%Z, jump 7 []); (1%Z, jump 8 [])])))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 12 Value (add_tagged (Var 2) (tagged 1))
             (Let 13 Value (block_of_one (Var 12))
                (Apply_cont (jump 2 [Var 13])))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata [])
             (Let 14 Value (add_tagged (Var 2) (tagged 2))
                (Let 15 Value (block_of_two 1 (Var 2) (Var 14))
                   (Apply_cont (jump 2 [Var 15])))))
          (Let_cont_non_recursive 5
             (Handler [] (make_handler_metadata [])
                (Apply_cont (jump 4 [])))
             (Let_cont_non_recursive 6
                (Handler [] (make_handler_metadata [])
                   (Apply_cont (jump 3 [])))
                (Let 16 naked_immediate (untag (Var 1))
                   (Switch (Var 16)
                      [(0%Z, jump 5 []); (1%Z, jump 6 [])])))))).

Definition either_heap_after : expr :=
  Let_cont_non_recursive 2
    (Handler [4; 7; 8; 10]
       (make_handler_metadata [naked_immediate; Value; Value; Value])
       (Let_cont_non_recursive 7
          (Handler [] (make_handler_metadata [])
             (Let 11 Value (add_tagged (Var 10) (tagged 3))
                (Apply_cont (jump 1 [Var 11]))))
          (Let_cont_non_recursive 8
             (Handler [] (make_handler_metadata [])
                (Let 9 Value (add_tagged (Var 8) (Var 7))
                   (Apply_cont (jump 1 [Var 9]))))
             (Switch (Var 4) [(0%Z, jump 7 []); (1%Z, jump 8 [])]))))
    (Let_cont_non_recursive 4
       (Handler [] (make_handler_metadata [])
          (Let 14 Value (add_tagged (Var 2) (tagged 2))
             (Apply_cont (jump 2 [naked 1; Var 14; Var 2; poison_value]))))
       (Let_cont_non_recursive 3
          (Handler [] (make_handler_metadata [])
             (Let 12 Value (add_tagged (Var 2) (tagged 1))
                (Apply_cont
                   (jump 2 [naked 0; poison_value; poison_value; Var 12]))))
          (Let 16 naked_immediate (untag (Var 1))
             (Switch (Var 16) [(0%Z, jump 4 []); (1%Z, jump 3 [])])))).

Definition either_heap_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_variants either_heap_before)
    either_heap_after
  = true.

Definition either_heap_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before either_heap_before)
    either_heap_after
  = false.

Definition either_heap_refines_statement : Prop :=
  expr_refines either_heap_before either_heap_after.

(* ======================================================================== *)
(* An option of a boxed number                                              *)
(* ======================================================================== *)

Definition poison_int64 : simple := Const (Poison naked_int64).

Definition int64_option_heap_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 4 Value (field 0 3)
                (Let 12 naked_int64 (unbox_int64 (Var 4))
                   (Let 13 naked_int64
                      (int64_arith Add (Var 12) (Const (Naked_int64 2)))
                      (Let 5 Value (box_int64 (Var 13))
                         (Apply_cont (jump 1 [Var 5])))))))
          (Let 6 naked_immediate (is_int (Var 3))
             (Let 7 Value (tag_immediate (Var 6))
                (Let_cont_non_recursive 6
                   (Handler [] (make_handler_metadata [])
                      (Apply_cont (jump 5 [])))
                   (Let 8 naked_immediate (untag (Var 7))
                      (Switch (Var 8)
                         [(0%Z, jump 6 []); (1%Z, jump 1 [Symbol 1])])))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 14 naked_int64 (unbox_int64 (Var 2))
             (Let 15 naked_int64
                (int64_arith Add (Var 14) (Const (Naked_int64 1)))
                (Let 9 Value (box_int64 (Var 15))
                   (Let 10 Value (block_of_one (Var 9))
                      (Apply_cont (jump 2 [Var 10])))))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata []) (Apply_cont (jump 3 [])))
          (Let 11 naked_immediate (untag (Var 1))
             (Switch (Var 11) [(0%Z, jump 2 [tagged 0]); (1%Z, jump 4 [])])))).

Definition int64_option_heap_after : expr :=
  Let_cont_non_recursive 2
    (Handler [6; 12] (make_handler_metadata [naked_immediate; naked_int64])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 13 naked_int64
                (int64_arith Add (Var 12) (Const (Naked_int64 2)))
                (Let 5 Value (box_int64 (Var 13))
                   (Apply_cont (jump 1 [Var 5])))))
          (Switch (Var 6) [(0%Z, jump 5 []); (1%Z, jump 1 [Symbol 1])])))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 14 naked_int64 (unbox_int64 (Var 2))
             (Let 15 naked_int64
                (int64_arith Add (Var 14) (Const (Naked_int64 1)))
                (Apply_cont (jump 2 [naked 0; Var 15])))))
       (Let 11 naked_immediate (untag (Var 1))
          (Switch (Var 11)
             [(0%Z, jump 2 [naked 1; poison_int64]); (1%Z, jump 3 [])]))).

Definition int64_option_heap_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_variants int64_option_heap_before)
    int64_option_heap_after
  = true.

Definition int64_option_heap_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before int64_option_heap_before)
    int64_option_heap_after
  = false.

Definition int64_option_heap_refines_statement : Prop :=
  expr_refines int64_option_heap_before int64_option_heap_after.

(* ======================================================================== *)
(* The bodies of the other file                                             *)
(* ======================================================================== *)

Definition earlier_examples_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_variants pair_computed_before)
    pair_computed_after
  = true
  ∧ expr_eqb [] [] (iterate 5 round_with_variants int64_built_before)
      int64_built_after
    = true.

(* ======================================================================== *)
(* The round of the validation                                              *)
(* ======================================================================== *)

(* Five rounds of FlambdaValidateSpec.normalise_expr_params, which has
   the pass, take the body before to the body after, up to bound
   names. *)
Definition validated_by_first_phase (before after : expr) : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_params before) after = true.

Definition bodies_validated_by_first_phase_statement : Prop :=
  validated_by_first_phase option_heap_before option_heap_after
  ∧ validated_by_first_phase list_heap_before list_heap_after
  ∧ validated_by_first_phase shape_heap_before shape_heap_after
  ∧ validated_by_first_phase shape2_heap_before shape2_heap_after
  ∧ validated_by_first_phase nested_heap_before nested_heap_after
  ∧ validated_by_first_phase either_heap_before either_heap_after
  ∧ validated_by_first_phase int64_option_heap_before int64_option_heap_after
  ∧ validated_by_first_phase pair_computed_before pair_computed_after
  ∧ validated_by_first_phase int64_built_before int64_built_after
  ∧ validated_by_first_phase pair_built_before pair_built_after.

(* ======================================================================== *)
(* A number that was tagged                                                 *)
(* ======================================================================== *)

Definition less_tagged (a b : simple) : named :=
  Prim
    (Binary
       (Int_comp Standard_tagged_immediate (Yielding_bool (Int_lt Signed))))
    [a; b].

(* The continuation of the join is 3 and its parameters are 4, the
   boolean, 5 and 6; that of the second test is 4, with the parameters
   12 and 13; that of the body of the loop is 5, with the parameters 7
   and 8, and 6 and 7 are the continuations of the arms that go to the
   body and to the second test. *)
Definition while_two_tests_before : expr :=
  Let_cont_non_recursive 3
    (Handler [4; 5; 6] (make_handler_metadata [Value; Value; Value])
       (Let_cont_non_recursive 5
          (Handler [7; 8] (make_handler_metadata [Value; Value])
             (Let 9 Value (add_tagged (Var 8) (Var 7))
                (Let 10 Value (add_tagged (Var 7) (tagged 1))
                   (Apply_cont (jump 2 [Var 10; Var 9])))))
          (Let_cont_non_recursive 6
             (Handler [] (make_handler_metadata [])
                (Apply_cont (jump 5 [Var 5; Var 6])))
             (Let 11 naked_immediate (untag (Var 4))
                (Switch (Var 11)
                   [(0%Z, jump 1 [Var 6]); (1%Z, jump 6 [])])))))
    (Let_cont_non_recursive 4
       (Handler [12; 13] (make_handler_metadata [Value; Value])
          (Let 14 naked_immediate (less_tagged (Var 13) (tagged 20))
             (Let 15 Value (tag_immediate (Var 14))
                (Apply_cont (jump 3 [Var 15; Var 12; Var 13])))))
       (Let 16 naked_immediate (less_tagged (Var 1) (Var 3))
          (Let 17 Value (tag_immediate (Var 16))
             (Let_cont_non_recursive 7
                (Handler [] (make_handler_metadata [])
                   (Apply_cont (jump 4 [Var 1; Var 2])))
                (Let 18 naked_immediate (untag (Var 17))
                   (Switch (Var 18)
                      [(0%Z, jump 3 [tagged 0; Var 1; Var 2]);
                       (1%Z, jump 7 [])])))))).

Definition while_two_tests_after : expr :=
  Let_cont_non_recursive 3
    (Handler [4] (make_handler_metadata [naked_immediate])
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 9 Value (add_tagged (Var 2) (Var 1))
                (Let 10 Value (add_tagged (Var 1) (tagged 1))
                   (Apply_cont (jump 2 [Var 10; Var 9])))))
          (Switch (Var 4) [(0%Z, jump 1 [Var 2]); (1%Z, jump 5 [])])))
    (Let 16 naked_immediate (less_tagged (Var 1) (Var 3))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata [])
             (Let 14 naked_immediate (less_tagged (Var 2) (tagged 20))
                (Apply_cont (jump 3 [Var 14]))))
          (Switch (Var 16) [(0%Z, jump 3 [naked 0]); (1%Z, jump 4 [])]))).

(* Five rounds of normalise_expr_params take the body before to the
   body after. *)
Definition while_two_tests_validated_statement : Prop :=
  validated_by_first_phase while_two_tests_before while_two_tests_after.

Definition while_two_tests_refines_statement : Prop :=
  expr_refines while_two_tests_before while_two_tests_after.

(* ======================================================================== *)
(* A symbol where there is nothing to untag                                 *)
(* ======================================================================== *)

(* The continuation of the join is 2 and its parameter is 3; 3 is the
   continuation that loads the field and 4 the one that untags the
   constant constructor. *)
Definition shape_constant_before : expr :=
  Let_static [Bind_symbol 1 (Static_const_block 0 [tagged 8])]
    (Let_cont_non_recursive 2
       (Handler [3] (make_handler_metadata [Value])
          (Let_cont_non_recursive 3
             (Handler [] (make_handler_metadata [])
                (Let 4 Value (field 0 3) (Apply_cont (jump 1 [Var 4]))))
             (Let_cont_non_recursive 4
                (Handler [] (make_handler_metadata [])
                   (Let 5 naked_immediate (untag (Var 3))
                      (Switch (Var 5)
                         [(0%Z, jump 1 [tagged 1]);
                          (1%Z, jump 1 [tagged 2])])))
                (Let 6 naked_immediate (is_int (Var 3))
                   (Switch (Var 6) [(0%Z, jump 3 []); (1%Z, jump 4 [])])))))
       (Let 7 naked_immediate (untag (Var 1))
          (Switch (Var 7)
             [(0%Z, jump 2 [tagged 0]); (1%Z, jump 2 [Symbol 1])]))).

Definition shape_constant_after : expr :=
  Let_static [Bind_symbol 1 (Static_const_block 0 [tagged 8])]
    (Let_cont_non_recursive 2
       (Handler [6; 4] (make_handler_metadata [naked_immediate; Value])
          (Switch (Var 6)
             [(0%Z, jump 1 [Var 4]); (1%Z, jump 1 [tagged 1])]))
       (Let 7 naked_immediate (untag (Var 1))
          (Switch (Var 7)
             [(0%Z, jump 2 [naked 1; poison_value]);
              (1%Z, jump 2 [naked 0; tagged 8])]))).

Definition shape_constant_validated_statement : Prop :=
  validated_by_first_phase shape_constant_before shape_constant_after.

Definition shape_constant_refines_statement : Prop :=
  expr_refines shape_constant_before shape_constant_after.

(* ======================================================================== *)
(* A constant block in code and in a loop                                   *)
(* ======================================================================== *)

(* The continuation of the join is 2 and its parameter is 3; 3 is the
   continuation that loads the field. *)
Definition constant_join_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 3
          (Handler [] (make_handler_metadata [])
             (Let 4 Value (field 0 3)
                (Let 5 Value (add_tagged (Var 2) (Var 4))
                   (Apply_cont (jump 1 [Var 5])))))
          (Let 6 naked_immediate (is_int (Var 3))
             (Switch (Var 6) [(0%Z, jump 3 []); (1%Z, jump 1 [Var 2])]))))
    (Let 7 naked_immediate (untag (Var 1))
       (Switch (Var 7)
          [(0%Z, jump 2 [tagged 0]); (1%Z, jump 2 [Symbol 1])])).

Definition constant_join_after : expr :=
  Let_cont_non_recursive 2
    (Handler [6; 4] (make_handler_metadata [naked_immediate; Value])
       (Let_cont_non_recursive 3
          (Handler [] (make_handler_metadata [])
             (Let 5 Value (add_tagged (Var 2) (Var 4))
                (Apply_cont (jump 1 [Var 5]))))
          (Switch (Var 6) [(0%Z, jump 3 []); (1%Z, jump 1 [Var 2])])))
    (Let 7 naked_immediate (untag (Var 1))
       (Switch (Var 7)
          [(0%Z, jump 2 [naked 1; poison_value]);
           (1%Z, jump 2 [naked 0; tagged 8])])).

Definition constant_block : static_const := Static_const_block 0 [tagged 8].

Definition constant_code (body : expr) : code :=
  Code [1; 2] 3 1 2
    (make_code_metadata [Singleton Value; Singleton Value] [Singleton Value])
    body.

Definition constant_in_code_before : expr :=
  Let_static
    [Bind_symbol 1 constant_block;
     Bind_code 1 (constant_code constant_join_before)]
    (Apply_cont (jump 1 [tagged 0])).

Definition constant_in_code_after : expr :=
  Let_static
    [Bind_symbol 1 constant_block;
     Bind_code 1 (constant_code constant_join_after)]
    (Apply_cont (jump 1 [tagged 0])).

Definition constant_below_code : expr :=
  Let_static [Bind_code 1 (constant_code constant_join_before)]
    (Let_static [Bind_symbol 1 constant_block]
       (Apply_cont (jump 1 [tagged 0]))).

(* The loop is the continuation 5. *)
Definition constant_in_loop_before : expr :=
  Let_static [Bind_symbol 1 constant_block]
    (Let_cont_recursive
       [(5,
         Handler [1; 2] (make_handler_metadata [Value; Value])
           constant_join_before)]
       (Apply_cont (jump 5 [Var 1; Var 2]))).

Definition constant_in_loop_after : expr :=
  Let_static [Bind_symbol 1 constant_block]
    (Let_cont_recursive
       [(5,
         Handler [1; 2] (make_handler_metadata [Value; Value])
           constant_join_after)]
       (Apply_cont (jump 5 [Var 1; Var 2]))).

(* Five rounds of FlambdaValidateSpec.normalise_params_round (the round
   of the first phase for a whole term, without the step that follows
   it in first_phase_round) take the term before to the term after, up
   to bound names. *)
Definition validated_by_rounds (before after : expr) : Prop :=
  expr_eqb [] [] (iterate 5 normalise_params_round before) after = true.

Definition constant_in_code_validated_statement : Prop :=
  validated_by_rounds constant_in_code_before constant_in_code_after.

Definition constant_in_code_refines_statement : Prop :=
  expr_refines constant_in_code_before constant_in_code_after.

Definition constant_below_code_left_statement : Prop :=
  validated_by_rounds constant_below_code constant_below_code.

Definition constant_in_loop_validated_statement : Prop :=
  validated_by_rounds constant_in_loop_before constant_in_loop_after.

Definition constant_in_loop_refines_statement : Prop :=
  expr_refines constant_in_loop_before constant_in_loop_after.

(* ======================================================================== *)
(* Given by the box, not by the path                                        *)
(* ======================================================================== *)

Definition pair_of (first second : simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values 0))) [first; second].

Definition poison_immediate : simple := Const (Poison naked_immediate).

(* The continuation that loads the first field is 3; 4 and 5 are those
   of the two arms. *)
Definition pair_constant_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let 4 Value (field 1 3)
          (Let 5 naked_immediate (untag (Var 4))
             (Let_cont_non_recursive 3
                (Handler [] (make_handler_metadata [])
                   (Let 6 Value (field 0 3) (Apply_cont (jump 1 [Var 6]))))
                (Switch (Var 5)
                   [(0%Z, jump 1 [tagged (-7)]); (1%Z, jump 3 [])])))))
    (Let_cont_non_recursive 4
       (Handler [] (make_handler_metadata [])
          (Let 9 Value (pair_of (Var 2) (tagged 1))
             (Apply_cont (jump 2 [Var 9]))))
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 10 Value (pair_of (tagged 0) (tagged 0))
                (Apply_cont (jump 2 [Var 10]))))
          (Let 11 naked_immediate (untag (Var 1))
             (Switch (Var 11) [(0%Z, jump 5 []); (1%Z, jump 4 [])])))).

Definition pair_constant_after : expr :=
  Let_cont_non_recursive 2
    (Handler [5; 6] (make_handler_metadata [naked_immediate; Value])
       (Switch (Var 5)
          [(0%Z, jump 1 [tagged (-7)]); (1%Z, jump 1 [Var 6])]))
    (Let 11 naked_immediate (untag (Var 1))
       (Switch (Var 11)
          [(0%Z, jump 2 [naked 0; tagged 0]);
           (1%Z, jump 2 [naked 1; Var 2])])).

Definition pair_constant_validated_statement : Prop :=
  validated_by_first_phase pair_constant_before pair_constant_after.

Definition pair_constant_refines_statement : Prop :=
  expr_refines pair_constant_before pair_constant_after.

(* The continuation of the arm of [Some] in the handler is 3, and 4 the
   one that loads the first field of the pair; 5 makes the option. *)
Definition option_pair_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let 6 naked_immediate (is_int (Var 3))
          (Let_cont_non_recursive 3
             (Handler [] (make_handler_metadata [])
                (Let 4 Value (field 0 3)
                   (Let 7 Value (field 1 4)
                      (Let 8 naked_immediate (untag (Var 7))
                         (Let_cont_non_recursive 4
                            (Handler [] (make_handler_metadata [])
                               (Let 9 Value (field 0 4)
                                  (Apply_cont (jump 1 [Var 9]))))
                            (Switch (Var 8)
                               [(0%Z, jump 1 [tagged 0]);
                                (1%Z, jump 4 [])]))))))
             (Switch (Var 6)
                [(0%Z, jump 3 []); (1%Z, jump 1 [tagged (-1)])]))))
    (Let_cont_non_recursive 5
       (Handler [] (make_handler_metadata [])
          (Let 10 Value (pair_of (Var 2) (tagged 1))
             (Let 11 Value (block_of_one (Var 10))
                (Apply_cont (jump 2 [Var 11])))))
       (Let 12 naked_immediate (untag (Var 1))
          (Switch (Var 12) [(0%Z, jump 2 [tagged 0]); (1%Z, jump 5 [])]))).

Definition option_pair_after : expr :=
  Let_cont_non_recursive 2
    (Handler [6; 8; 9]
       (make_handler_metadata [naked_immediate; naked_immediate; Value])
       (Let_cont_non_recursive 3
          (Handler [] (make_handler_metadata [])
             (Switch (Var 8)
                [(0%Z, jump 1 [tagged 0]); (1%Z, jump 1 [Var 9])]))
          (Switch (Var 6)
             [(0%Z, jump 3 []); (1%Z, jump 1 [tagged (-1)])])))
    (Let 12 naked_immediate (untag (Var 1))
       (Switch (Var 12)
          [(0%Z, jump 2 [naked 1; poison_immediate; poison_value]);
           (1%Z, jump 2 [naked 0; naked 1; Var 2])])).

Definition option_pair_validated_statement : Prop :=
  validated_by_first_phase option_pair_before option_pair_after.

Definition option_pair_refines_statement : Prop :=
  expr_refines option_pair_before option_pair_after.

(* The continuation of the arm of [Some] in the handler is 3; 5 makes
   the option. *)
Definition option_fields_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let 6 naked_immediate (is_int (Var 3))
          (Let_cont_non_recursive 3
             (Handler [] (make_handler_metadata [])
                (Let 4 Value (field 0 3)
                   (Let 7 Value (field 0 4)
                      (Let 8 Value (field 1 4)
                         (Let 9 Value (add_tagged (Var 8) (Var 7))
                            (Apply_cont (jump 1 [Var 9])))))))
             (Switch (Var 6)
                [(0%Z, jump 3 []); (1%Z, jump 1 [tagged 0])]))))
    (Let_cont_non_recursive 5
       (Handler [] (make_handler_metadata [])
          (Let 10 Value (pair_of (tagged 3) (Var 2))
             (Let 11 Value (block_of_one (Var 10))
                (Apply_cont (jump 2 [Var 11])))))
       (Let 12 naked_immediate (untag (Var 1))
          (Switch (Var 12) [(0%Z, jump 2 [tagged 0]); (1%Z, jump 5 [])]))).

Definition option_fields_after : expr :=
  Let_cont_non_recursive 2
    (Handler [6; 8; 7] (make_handler_metadata [naked_immediate; Value; Value])
       (Let_cont_non_recursive 3
          (Handler [] (make_handler_metadata [])
             (Let 9 Value (add_tagged (Var 8) (Var 7))
                (Apply_cont (jump 1 [Var 9]))))
          (Switch (Var 6) [(0%Z, jump 3 []); (1%Z, jump 1 [tagged 0])])))
    (Let 12 naked_immediate (untag (Var 1))
       (Switch (Var 12)
          [(0%Z, jump 2 [naked 1; poison_value; poison_value]);
           (1%Z, jump 2 [naked 0; Var 2; tagged 3])])).

Definition option_fields_validated_statement : Prop :=
  validated_by_first_phase option_fields_before option_fields_after.

Definition option_fields_refines_statement : Prop :=
  expr_refines option_fields_before option_fields_after.

(* ======================================================================== *)
(* A load to which blocks of two tags come                                  *)
(* ======================================================================== *)

Definition block_tagged (tag : Z) (contents : simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values tag))) [contents].

(* The two arms, which make the blocks and give them to the join. *)
Definition two_tags_arms : expr :=
  Let_cont_non_recursive 4
    (Handler [] (make_handler_metadata [])
       (Let 9 Value (block_tagged 0 (Var 2)) (Apply_cont (jump 2 [Var 9]))))
    (Let_cont_non_recursive 5
       (Handler [] (make_handler_metadata [])
          (Let 10 Value (block_tagged 1 (Var 2))
             (Apply_cont (jump 2 [Var 10]))))
       (Let 11 naked_immediate (untag (Var 1))
          (Switch (Var 11) [(0%Z, jump 5 []); (1%Z, jump 4 [])]))).

(* The continuation that loads the field is 3. *)
Definition two_tags_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let 6 naked_immediate (is_int (Var 3))
          (Let_cont_non_recursive 3
             (Handler [] (make_handler_metadata [])
                (Let 7 Value (field 0 3)
                   (Let 8 Value (add_tagged (Var 7) (tagged 1))
                      (Apply_cont (jump 1 [Var 8])))))
             (Switch (Var 6)
                [(0%Z, jump 3 []); (1%Z, jump 1 [tagged 0])]))))
    two_tags_arms.

Definition two_tags_after : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let 7 Value (field 0 3)
          (Let 8 Value (add_tagged (Var 7) (tagged 1))
             (Apply_cont (jump 1 [Var 8])))))
    two_tags_arms.

Definition two_tags_validated_statement : Prop :=
  validated_by_first_phase two_tags_before two_tags_after.

Definition two_tags_refines_statement : Prop :=
  expr_refines two_tags_before two_tags_after.

(* ======================================================================== *)
(* A box that the term has already read                                     *)
(* ======================================================================== *)

Definition less_int64 (a b : simple) : named :=
  Prim
    (Binary (Int_comp Standard_naked_int64 (Yielding_bool (Int_lt Signed))))
    [a; b].

Definition int64_max_before : expr :=
  Let 4 naked_int64 (unbox_int64 (Var 2))
    (Let 5 naked_int64 (unbox_int64 (Var 1))
       (Let 6 naked_immediate (less_int64 (Var 5) (Var 4))
          (Let_cont_non_recursive 2
             (Handler [3] (make_handler_metadata [Value])
                (Let 7 naked_int64 (unbox_int64 (Var 3))
                   (Let 9 naked_int64
                      (int64_arith Add (Var 7) (Const (Naked_int64 1)))
                      (Let 10 Value (box_int64 (Var 9))
                         (Apply_cont (jump 1 [Var 10]))))))
             (Switch (Var 6)
                [(0%Z, jump 2 [Var 1]); (1%Z, jump 2 [Var 2])])))).

Definition int64_max_after : expr :=
  Let 4 naked_int64 (unbox_int64 (Var 2))
    (Let 5 naked_int64 (unbox_int64 (Var 1))
       (Let 6 naked_immediate (less_int64 (Var 5) (Var 4))
          (Let_cont_non_recursive 2
             (Handler [7] (make_handler_metadata [naked_int64])
                (Let 9 naked_int64
                   (int64_arith Add (Var 7) (Const (Naked_int64 1)))
                   (Let 10 Value (box_int64 (Var 9))
                      (Apply_cont (jump 1 [Var 10])))))
             (Switch (Var 6)
                [(0%Z, jump 2 [Var 5]); (1%Z, jump 2 [Var 4])])))).

Definition int64_max_validated_statement : Prop :=
  validated_by_first_phase int64_max_before int64_max_after.

Definition int64_max_refines_statement : Prop :=
  expr_refines int64_max_before int64_max_after.

(* ======================================================================== *)
(* What a handler reads in two places                                       *)
(* ======================================================================== *)

Definition mul_tagged (a b : simple) : named :=
  Prim (Binary (Int_arith Standard_tagged_immediate Mul)) [a; b].

Definition matched_twice_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let_cont_non_recursive 3
          (Handler [4] (make_handler_metadata [Value])
             (Let_cont_non_recursive 4
                (Handler [5] (make_handler_metadata [Value])
                   (Let 15 Value (add_tagged (Var 4) (Var 5))
                      (Apply_cont (jump 1 [Var 15]))))
                (Let_cont_non_recursive 5
                   (Handler [] (make_handler_metadata [])
                      (Let 13 Value (field 0 3)
                         (Let 14 Value (mul_tagged (Var 13) (tagged 2))
                            (Apply_cont (jump 4 [Var 14])))))
                   (Let_cont_non_recursive 6
                      (Handler [] (make_handler_metadata [])
                         (Apply_cont (jump 5 [])))
                      (Let 10 naked_immediate (is_int (Var 3))
                         (Let 11 Value (tag_immediate (Var 10))
                            (Let 12 naked_immediate (untag (Var 11))
                               (Switch (Var 12)
                                  [(0%Z, jump 6 []);
                                   (1%Z, jump 4 [tagged 10])]))))))))
          (Let_cont_non_recursive 7
             (Handler [] (make_handler_metadata [])
                (Let 9 Value (field 0 3) (Apply_cont (jump 3 [Var 9]))))
             (Let_cont_non_recursive 8
                (Handler [] (make_handler_metadata [])
                   (Apply_cont (jump 7 [])))
                (Let 6 naked_immediate (is_int (Var 3))
                   (Let 7 Value (tag_immediate (Var 6))
                      (Let 8 naked_immediate (untag (Var 7))
                         (Switch (Var 8)
                            [(0%Z, jump 8 []);
                             (1%Z, jump 3 [tagged 0])]))))))))
    (Let_cont_non_recursive 9
       (Handler [] (make_handler_metadata [])
          (Let 16 Value (add_tagged (Var 2) (tagged 1))
             (Let 17 Value (block_of_one (Var 16))
                (Apply_cont (jump 2 [Var 17])))))
       (Let_cont_non_recursive 10
          (Handler [] (make_handler_metadata []) (Apply_cont (jump 9 [])))
          (Let 18 naked_immediate (untag (Var 1))
             (Switch (Var 18)
                [(0%Z, jump 2 [tagged 0]); (1%Z, jump 10 [])])))).

Definition matched_twice_after : expr :=
  Let_cont_non_recursive 2
    (Handler [6; 9] (make_handler_metadata [naked_immediate; Value])
       (Let_cont_non_recursive 3
          (Handler [4] (make_handler_metadata [Value])
             (Let_cont_non_recursive 4
                (Handler [5] (make_handler_metadata [Value])
                   (Let 15 Value (add_tagged (Var 4) (Var 5))
                      (Apply_cont (jump 1 [Var 15]))))
                (Let_cont_non_recursive 5
                   (Handler [] (make_handler_metadata [])
                      (Let 14 Value (mul_tagged (Var 9) (tagged 2))
                         (Apply_cont (jump 4 [Var 14]))))
                   (Switch (Var 6)
                      [(0%Z, jump 5 []); (1%Z, jump 4 [tagged 10])]))))
          (Switch (Var 6) [(0%Z, jump 3 [Var 9]); (1%Z, jump 3 [tagged 0])])))
    (Let_cont_non_recursive 9
       (Handler [] (make_handler_metadata [])
          (Let 16 Value (add_tagged (Var 2) (tagged 1))
             (Apply_cont (jump 2 [naked 0; Var 16]))))
       (Let 18 naked_immediate (untag (Var 1))
          (Switch (Var 18)
             [(0%Z, jump 2 [naked 1; poison_value]); (1%Z, jump 9 [])]))).

Definition matched_twice_validated_statement : Prop :=
  validated_by_first_phase matched_twice_before matched_twice_after.

Definition matched_twice_refines_statement : Prop :=
  expr_refines matched_twice_before matched_twice_after.

(* ======================================================================== *)
(* A loop that is given a pair at every turn                                *)
(* ======================================================================== *)

Definition pair_loop_before : expr :=
  Let 5 Value (pair_of (tagged 0) (tagged 1))
    (Let_cont_recursive
       [(2,
         Handler [3; 4] (make_handler_metadata [Value; Value])
           (Let 6 Value (field 1 4)
              (Let 7 Value (field 0 4)
                 (Let 8 Value (add_tagged (Var 7) (Var 6))
                    (Let 9 Value (pair_of (Var 6) (Var 8))
                       (Let 10 Value (add_tagged (Var 3) (tagged 1))
                          (Let 11 naked_immediate
                             (less_tagged (Var 10) (Var 1))
                             (Switch (Var 11)
                                [(0%Z, jump 1 [Var 6]);
                                 (1%Z, jump 2 [Var 10; Var 9])]))))))))]
       (Apply_cont (jump 2 [tagged 1; Var 5]))).

Definition pair_loop_after : expr :=
  Let_cont_recursive
    [(2,
      Handler [3; 6; 7] (make_handler_metadata [Value; Value; Value])
        (Let 8 Value (add_tagged (Var 7) (Var 6))
           (Let 10 Value (add_tagged (Var 3) (tagged 1))
              (Let 11 naked_immediate (less_tagged (Var 10) (Var 1))
                 (Switch (Var 11)
                    [(0%Z, jump 1 [Var 6]);
                     (1%Z, jump 2 [Var 10; Var 8; Var 6])])))))]
    (Apply_cont (jump 2 [tagged 1; tagged 1; tagged 0])).

Definition pair_loop_validated_statement : Prop :=
  validated_by_first_phase pair_loop_before pair_loop_after.

Definition pair_loop_refines_statement : Prop :=
  expr_refines pair_loop_before pair_loop_after.

(* ======================================================================== *)
(* A box that one join passes on to another                                 *)
(* ======================================================================== *)

Definition int64_passed_on_body_before : expr :=
  Let_cont_non_recursive 3
    (Handler [5] (make_handler_metadata [Value])
       (Let_cont_non_recursive 4
          (Handler [6] (make_handler_metadata [Value])
             (Let 7 naked_int64 (unbox_int64 (Var 6))
                (Let 8 naked_int64 (unbox_int64 (Var 6))
                   (Let 9 naked_int64 (int64_arith Add (Var 8) (Var 7))
                      (Let 10 Value (box_int64 (Var 9))
                         (Apply_cont (jump 1 [Var 10])))))))
          (Let 11 naked_immediate (untag (Var 1))
             (Switch (Var 11)
                [(0%Z, jump 4 [Symbol 1]); (1%Z, jump 4 [Var 5])]))))
    (Let_cont_non_recursive 5
       (Handler [] (make_handler_metadata [])
          (Let 12 naked_int64 (unbox_int64 (Var 3))
             (Let 13 naked_int64
                (int64_arith Add (Var 12) (Const (Naked_int64 1)))
                (Let 14 Value (box_int64 (Var 13))
                   (Apply_cont (jump 3 [Var 14]))))))
       (Let_cont_non_recursive 6
          (Handler [] (make_handler_metadata [])
             (Let 15 naked_int64 (unbox_int64 (Var 3))
                (Let 16 naked_int64
                   (int64_arith Add (Var 15) (Const (Naked_int64 2)))
                   (Let 17 Value (box_int64 (Var 16))
                      (Apply_cont (jump 3 [Var 17]))))))
          (Let_cont_non_recursive 7
             (Handler [] (make_handler_metadata []) (Apply_cont (jump 5 [])))
             (Let_cont_non_recursive 8
                (Handler [] (make_handler_metadata [])
                   (Apply_cont (jump 6 [])))
                (Let 18 naked_immediate (untag (Var 2))
                   (Switch (Var 18)
                      [(0%Z, jump 8 []); (1%Z, jump 7 [])])))))).

Definition int64_passed_on_body_after : expr :=
  Let_cont_non_recursive 3
    (Handler [5] (make_handler_metadata [naked_int64])
       (Let_cont_non_recursive 4
          (Handler [6] (make_handler_metadata [naked_int64])
             (Let 9 naked_int64 (int64_arith Add (Var 6) (Var 6))
                (Let 10 Value (box_int64 (Var 9))
                   (Apply_cont (jump 1 [Var 10])))))
          (Let 11 naked_immediate (untag (Var 1))
             (Switch (Var 11)
                [(0%Z, jump 4 [Const (Naked_int64 3)]);
                 (1%Z, jump 4 [Var 5])]))))
    (Let_cont_non_recursive 5
       (Handler [] (make_handler_metadata [])
          (Let 12 naked_int64 (unbox_int64 (Var 3))
             (Let 13 naked_int64
                (int64_arith Add (Var 12) (Const (Naked_int64 1)))
                (Apply_cont (jump 3 [Var 13])))))
       (Let_cont_non_recursive 6
          (Handler [] (make_handler_metadata [])
             (Let 15 naked_int64 (unbox_int64 (Var 3))
                (Let 16 naked_int64
                   (int64_arith Add (Var 15) (Const (Naked_int64 2)))
                   (Apply_cont (jump 3 [Var 16])))))
          (Let 18 naked_immediate (untag (Var 2))
             (Switch (Var 18) [(0%Z, jump 6 []); (1%Z, jump 5 [])])))).

Definition boxed_three : static_const :=
  Static_const_boxed_number (Const (Naked_int64 3)).

Definition passed_on_code (body : expr) : code :=
  Code [1; 2; 3] 4 1 2
    (make_code_metadata [Singleton Value; Singleton Value; Singleton Value]
       [Singleton Value])
    body.

Definition int64_passed_on_before : expr :=
  Let_static
    [Bind_symbol 1 boxed_three;
     Bind_code 1 (passed_on_code int64_passed_on_body_before)]
    (Apply_cont (jump 1 [tagged 0])).

Definition int64_passed_on_after : expr :=
  Let_static
    [Bind_symbol 1 boxed_three;
     Bind_code 1 (passed_on_code int64_passed_on_body_after)]
    (Apply_cont (jump 1 [tagged 0])).

Definition int64_passed_on_validated_statement : Prop :=
  validated_by_rounds int64_passed_on_before int64_passed_on_after.

Definition int64_passed_on_refines_statement : Prop :=
  expr_refines int64_passed_on_before int64_passed_on_after.
