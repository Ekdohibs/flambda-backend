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

(* The relation of the proofs of FlambdaReaperUnboxingRefinesCallsSpec.v
   between the unit rebuilt with the change of calling conventions alone
   (the first run) and the unit rebuilt with unboxing in the mode of the
   fix or of its variant (the second run), and that it is a simulation
   ([uc_refines]).

   A parameter kept by the first run is, in the second ([pshape]): the
   same ([Psh_plain]); dead ([Psh_dead]), a variable with no use that is
   not unboxed, to which the second run may give anything of its kind
   (the poison of [Keep_unused] or of a wrapper); or unboxed
   ([Psh_unboxed]), the variables of its fields, each the value at its
   path ([fields_vals]). A continuation is described by the shapes of its
   parameters ([kshape]), and by whether it is no way out: a handler of
   the term, or the return continuation of code, which the first run
   applies as a closure. A call relates its arguments as the shapes of
   the parameters of its code say ([call_rel], [args_rel2]) and returns to
   a continuation that is no way out, a handler or, in tail position, the
   return continuation of the code it is in, whose shapes those of the
   results [transport] to, directly ([uc_call]), through a wrapper of the
   second run ([uc_call_wrap], [wrap_u_ok]) or through one of each run
   ([uc_call_wraps], [wrap_both_ok]). Through tail calls, the
   continuation of code is a handler under wrappers of either run, any
   number of them ([cont_rel_wrap_u], [cont_rel_wrap_both]), which
   [cont_answered] answers by induction. Code raises exceptions of one
   shape ([cs_exn]: plain, or dead where the second run raises poison),
   which a call gives to a way out, which takes them plain, to the
   exception continuation of the code it is in, or to a handler, whose
   other parameters take the extra arguments of the call
   ([exn_call_rel]); the exception continuation of the callee is that
   handler with the extra arguments ([cont_rel_extra]), which is no way
   out, so that a dead exception never leaves the unit. A call whose code
   is not known ([call_target]) gives the same arguments in both runs and
   is made at a shape that every code of the unit fits ([ind_targets],
   [ind_compat]): the code keeps its calling convention, so that the first
   run's code and the second's have the same arities ([code_frame],
   [cs_keeps]) and the call is in both a full call, an over-application
   or a partial application alike ([closure_call_answered]); what is left
   of an over-application is related ([cont_rel_over]) and called in turn
   as such a call. The states are related in the body of the unit
   ([related_main]), in the body of code called ([related_call], with the
   frame of the caller), or as a call returns ([related_pending]);
   [related2_answered] answers each step of the first run by steps of the
   second, so that the relation is a shrinking simulation. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Shapes                                                                   *)
(* ======================================================================== *)

(* What a parameter kept by the first run is in the second: the same
   ([Psh_plain]), the same but given anything of its kind, as it has no
   use ([Psh_dead]), or the variables of its unboxed fields
   ([Psh_unboxed]). *)
Inductive pshape : Type :=
  | Psh_plain
  | Psh_dead (kind : flambda_kind)
  | Psh_unboxed (fields : unboxed_fields).

(* What a continuation in scope takes: the shapes of the parameters that
   the first run keeps, their kinds in the first run, and whether it is
   no way out: a handler of the term, or the return continuation of code,
   which a call of the term gives a handler, through wrappers or not. *)
Record kshape : Type := Kshape {
  ks_shapes : list pshape;
  ks_kinds : list flambda_kind;
  ks_handler : bool
}.

(* The values of the leaves of [fields] in a value [w]. *)
Definition fields_vals (fields : unboxed_fields) (w : value) (ws : list value)
    : Prop :=
  Forall2
    (fun (pk : list field * flambda_kind) (w' : value) =>
       path_value (fst pk) w = Some w' ∧ kind_of_value w' = snd pk)
    (fields_paths fields) ws.

(* Values given to parameters of shapes [sh]: the first run's and the
   second's. *)
Inductive vrel : list pshape -> list value -> list value -> Prop :=
  | vrel_beyond (vs : list value) : vrel [] vs vs
  | vrel_missing (o : pshape) (sh : list pshape) : vrel (o :: sh) [] []
  | vrel_plain (sh : list pshape) (v : value) (vs vs' : list value) :
    vrel sh vs vs' -> vrel (Psh_plain :: sh) (v :: vs) (v :: vs')
  | vrel_dead (sh : list pshape) (kind : flambda_kind) (v v' : value)
      (vs vs' : list value) :
    (v' = v ∨ kind_of_value v' = kind) -> vrel sh vs vs' ->
    vrel (Psh_dead kind :: sh) (v :: vs) (v' :: vs')
  | vrel_unboxed (sh : list pshape) (fields : unboxed_fields) (v : value)
      (ws vs vs' : list value) :
    fields_vals fields v ws -> vrel sh vs vs' ->
    vrel (Psh_unboxed fields :: sh) (v :: vs) (ws ++ vs').

(* Values of the shape [sh] are of the shape [sh']: what a callee returns
   is what its continuation takes. *)
Inductive fits : pshape -> pshape -> Prop :=
  | fits_plain : fits Psh_plain Psh_plain
  | fits_plain_dead (kind : flambda_kind) : fits Psh_plain (Psh_dead kind)
  | fits_dead (kind : flambda_kind) : fits (Psh_dead kind) (Psh_dead kind)
  | fits_unboxed (fields fields' : unboxed_fields) :
    fields_paths fields = fields_paths fields' ->
    fits (Psh_unboxed fields) (Psh_unboxed fields').

(* Values of the shapes [sh], given to a continuation that takes the
   kinds [KS] in the first run, are of the shapes [sh']. *)
Definition transport (sh sh' : list pshape) (KS : list flambda_kind)
    : Prop :=
  ∀ vs vs' : list value,
    vrel sh vs vs' -> map kind_of_value vs = KS -> vrel sh' vs vs'.

(* What code takes and returns: the shapes of the parameters and of the
   results that the first run keeps, the kinds of those results in the
   first run, the arities of the code in the second, the shape of the
   exceptions that it raises, and whether it keeps its calling convention,
   the first run then giving it the arities of the second. *)
Record code_shape : Type := Code_shape {
  cs_params : list pshape;
  cs_results : list pshape;
  cs_result_kinds : list flambda_kind;
  cs_params_arity : arity;
  cs_result_arity : arity;
  cs_exn : pshape;
  cs_keeps : bool
}.

(* The kinds of the arguments in the second run, those of the first being
   [ks]. *)
Fixpoint shape_kinds (sh : list pshape) (ks : list flambda_kind)
    : list flambda_kind :=
  match sh, ks with
  | [], _ => ks
  | _, [] => []
  | Psh_plain :: sh, k :: ks => k :: shape_kinds sh ks
  | Psh_dead kind :: sh, _ :: ks => kind :: shape_kinds sh ks
  | Psh_unboxed fields :: sh, _ :: ks =>
    map snd (fields_paths fields) ++ shape_kinds sh ks
  end.

Definition no_dead (sh : list pshape) : Prop :=
  ∀ kind : flambda_kind, Psh_dead kind ∉ sh.

Definition no_unboxed (sh : list pshape) : Prop :=
  ∀ fields : unboxed_fields, Psh_unboxed fields ∉ sh.

(* A wrapper: a handler [w] that gives [k] the arguments [args], over
   [body]. *)
Definition wrap (w : continuation) (ps : list variable)
    (md : handler_metadata) (k : continuation) (args : list simple)
    (body : expr) : expr :=
  Let_cont_non_recursive w
    (Handler ps md
       (Apply_cont
          {| ac_continuation := k; ac_args := args; ac_trap_action := None |}))
    body.

Definition wrapper_closure (ps : list variable) (md : handler_metadata)
    (k : continuation) (args : list simple) (vars : env) (conts : cont_env)
    : cont_value :=
  Closure
    (Handler ps md
       (Apply_cont
          {| ac_continuation := k; ac_args := args; ac_trap_action := None |}))
    vars conts [].

(* A wrapper of the second run alone: given what the callee returns, of
   the shapes [sh_r], it gives the continuation, whose parameters are of
   the shapes [sh_k] and which takes [KS] in the first run, what the first
   run gives it directly. *)
Definition wrap_u_ok (sh_r sh_k : list pshape) (KS : list flambda_kind)
    (wps : list variable) (wmd : handler_metadata) (wargs : list simple)
    : Prop :=
  ∀ vs vs' : list value,
    vrel sh_r vs vs' -> map kind_of_value vs = KS ->
    kinds_eqb (map kind_of_value vs') (hm_params_kinds wmd) = true
    ∧ ∀ wv : env,
        ∃ (hv : env) (vs'' : list value),
          bind_params wps vs' wv = Some hv
          ∧ eval_simples hv wargs = Some vs'' ∧ vrel sh_k vs vs''.

(* Wrappers in both runs: given what the callee returns, of the shapes
   [sh_r], they give the continuation, whose parameters are of the shapes
   [sh_k], related arguments. *)
Definition wrap_both_ok (sh_r sh_k : list pshape) (cps : list variable)
    (cmd : handler_metadata) (cargs : list simple) (wps : list variable)
    (wmd : handler_metadata) (wargs : list simple) : Prop :=
  ∀ vs vs' : list value,
    vrel sh_r vs vs' ->
    kinds_eqb (map kind_of_value vs) (hm_params_kinds cmd) = true ->
    kinds_eqb (map kind_of_value vs') (hm_params_kinds wmd) = true
    ∧ ∀ (cv wv hv : env),
        bind_params cps vs cv = Some hv ->
        ∃ hv' : env,
          bind_params wps vs' wv = Some hv'
          ∧ ∀ ws : list value,
              eval_simples hv cargs = Some ws ->
              ∃ ws' : list value,
                eval_simples hv' wargs = Some ws' ∧ vrel sh_k ws ws'.

(* ======================================================================== *)
(* Terms related across calls                                               *)
(* ======================================================================== *)

Section Related.

(* The unboxed fields of the names, the parameters that the second run
   gives anything of their kind, and the shapes of the code. *)
Variable M : list (node * unboxed_fields).
Variable dead : list variable.
Variable CS : code_id -> option code_shape.

Definition plain2 (z : variable) : bool :=
  plain_variable M z && negb (among z dead).

Definition plain_simple2 (s : simple) : bool :=
  match s with
  | Var x => plain2 x
  | Symbol sym => negb (Keyed.is_bound node_eqb (Node_symbol sym) M)
  | Const _ => true
  end.

Definition named_plain2 (d : named) : bool :=
  match d with
  | Simple s => plain_simple2 s
  | Prim _ args => forallb plain_simple2 args
  | Closure_of_set _ _ value_slots =>
    forallb (fun ws => plain_simple2 (snd ws)) value_slots
  | _ => false
  end.

Definition local_binding2 (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ fields) =>
    forallb plain_simple2 fields
  | _ => false
  end.

Definition alloc_entry2 (x : variable) (d : named)
    (fu : field * unboxed_field)
    (l : list (variable * flambda_kind * simple)) : Prop :=
  match fu with
  | (f, Not_unboxed v) =>
    ∃ (kind : flambda_kind) (s : simple),
      l = [(v, kind, s)] ∧ alloc_field d f kind s ∧ plain_simple2 s = true
  | (f, Unboxed nested) =>
    ∃ (b : variable) (fields_b : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind)),
      alloc_field d f Value (Var b) ∧ fields_of M b = Some fields_b ∧ x <> b
      ∧ unboxed_field_pairs Value (Unboxed nested) (Unboxed fields_b)
        = Some pairs
      ∧ l = pairs_lets pairs
  end.

Inductive params_rel2 : list pshape -> list (variable * flambda_kind)
    -> list (variable * flambda_kind) -> Prop :=
  | params_rel2_nil : params_rel2 [] [] []
  | params_rel2_plain (sh : list pshape) (p : variable) (kind : flambda_kind)
      (ps ps' : list (variable * flambda_kind)) :
    plain2 p = true -> params_rel2 sh ps ps' ->
    params_rel2 (Psh_plain :: sh) ((p, kind) :: ps) ((p, kind) :: ps')
  | params_rel2_dead (sh : list pshape) (p : variable) (kind : flambda_kind)
      (ps ps' : list (variable * flambda_kind)) :
    among p dead = true -> fields_of M p = None ->
    among p (field_variables M) = false -> params_rel2 sh ps ps' ->
    params_rel2 (Psh_dead kind :: sh) ((p, kind) :: ps) ((p, kind) :: ps')
  | params_rel2_unboxed (sh : list pshape) (p : variable)
      (fields : unboxed_fields) (ps ps' : list (variable * flambda_kind)) :
    fields_of M p = Some fields -> among p (field_variables M) = false ->
    params_rel2 sh ps ps' ->
    params_rel2 (Psh_unboxed fields :: sh) ((p, Value) :: ps)
      (fields_leaves fields ++ ps').

Inductive args_rel2 : list pshape -> list simple -> list simple -> Prop :=
  | args_rel2_beyond (args : list simple) :
    forallb plain_simple2 args = true -> args_rel2 [] args args
  | args_rel2_missing (o : pshape) (sh : list pshape) :
    args_rel2 (o :: sh) [] []
  | args_rel2_plain (sh : list pshape) (a : simple)
      (args args' : list simple) :
    plain_simple2 a = true -> args_rel2 sh args args' ->
    args_rel2 (Psh_plain :: sh) (a :: args) (a :: args')
  | args_rel2_dead (sh : list pshape) (kind : flambda_kind) (a a' : simple)
      (args args' : list simple) :
    (a' = Const (Poison kind) ∨ (a' = a ∧ plain_simple2 a = true)) ->
    args_rel2 sh args args' ->
    args_rel2 (Psh_dead kind :: sh) (a :: args) (a' :: args')
  | args_rel2_unboxed (sh : list pshape) (fields fields_y : unboxed_fields)
      (y : variable) (pairs : list (variable * variable * flambda_kind))
      (args args' : list simple) :
    fields_of M y = Some fields_y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs ->
    args_rel2 sh args args' ->
    args_rel2 (Psh_unboxed fields :: sh) (Var y :: args)
      (map (fun '(_, x', _) => Var x') pairs ++ args').

Definition kshape_at (K : finite_map kshape) (k : continuation)
    : list pshape :=
  match lookup k K with
  | Some ks => ks_shapes ks
  | None => []
  end.

Definition ac_rel2 (K : finite_map kshape) (ac ac' : apply_cont) : Prop :=
  ac_continuation ac' = ac_continuation ac
  ∧ ac_trap_action ac' = ac_trap_action ac
  ∧ args_rel2 (kshape_at K (ac_continuation ac)) (ac_args ac) (ac_args ac').

(* Code of the shape [cs'] that a call whose code is not known, made at
   the shape [cs], may call: it keeps its calling convention, so that the
   two runs call it alike, whatever the number of arguments; its
   parameters are plain; the kinds of its results are those of its result
   arity; its exception fits that of the call; its results fit those of
   the call where they are of its return arity; and a single result of
   kind [Value], which an over-application calls, is plain. *)
Definition ind_compat (cs' cs : code_shape) : Prop :=
  cs_keeps cs' = true
  ∧ Forall (fun o => o = Psh_plain) (cs_params cs')
  ∧ cs_result_kinds cs' = unarize (cs_result_arity cs')
  ∧ fits (cs_exn cs') (cs_exn cs)
  ∧ (cs_result_arity cs' = cs_result_arity cs ->
     Forall2 fits (cs_results cs') (cs_results cs))
  ∧ (cs_result_arity cs' = [Singleton Value] -> cs_results cs' = [Psh_plain]).

(* The shape [cs] of a call whose code is not known: every code may be
   called at it; its parameters are plain, and its results are not
   unboxed and of the kinds of its return arity. *)
Definition ind_targets (cs : code_shape) : Prop :=
  (∀ (id : code_id) (cs' : code_shape), CS id = Some cs' -> ind_compat cs' cs)
  ∧ Forall (fun o => o = Psh_plain) (cs_params cs)
  ∧ no_unboxed (cs_results cs)
  ∧ cs_result_kinds cs = unarize (cs_result_arity cs).

(* The code that two calls call: the code [id] they are known to call,
   directly; or code that is not known, the two calls of the same kind,
   at the arities of the first. *)
Definition call_target (ap ap' : apply_expr) (cs : code_shape) : Prop :=
  (∃ id : code_id,
     ap_call_kind ap = Direct id ∧ ap_call_kind ap' = Direct id
     ∧ CS id = Some cs)
  ∨ (ap_call_kind ap' = ap_call_kind ap ∧ ind_targets cs
     ∧ ap_args_arity ap = cs_params_arity cs
     ∧ ap_return_arity ap = cs_result_arity cs).

(* What two calls share, but for where they return and the extra
   arguments of their exception continuation: the code they call, their
   callee, their arguments at the shapes of the parameters of the code,
   the arities of the second those of the code in the second, as many
   groups of arguments as the code has parameters, and the exception
   continuation. *)
Definition call_rel (ap ap' : apply_expr) (cs : code_shape) : Prop :=
  call_target ap ap' cs
    ∧ ap_callee ap' = ap_callee ap ∧ plain_simple2 (ap_callee ap) = true
    ∧ args_rel2 (cs_params cs) (ap_args ap) (ap_args ap')
    ∧ no_dead (cs_params cs)
    ∧ ap_args_arity ap' = cs_params_arity cs
    ∧ ap_return_arity ap' = cs_result_arity cs
    ∧ length (ap_args_arity ap) = length (cs_params_arity cs)
    ∧ unarize (cs_params_arity cs)
      = shape_kinds (cs_params cs) (unarize (ap_args_arity ap))
    ∧ ap_exn_continuation ap' = ap_exn_continuation ap.

(* The exception continuation of a call of code that raises exceptions of
   the shape [o], with the extra arguments of the two calls: a way out,
   which takes the exception plain, or the exception continuation of the
   code the call is in, which takes it alone, with no extra argument; or a
   handler, which is no way out, whose first parameter takes the
   exception and whose others the extra arguments, one each. *)
Definition exn_call_rel (K : finite_map kshape) (o : pshape)
    (ap ap' : apply_expr) : Prop :=
  (ap_exn_extra_args ap = [] ∧ ap_exn_extra_args ap' = []
   ∧ (lookup (ap_exn_continuation ap) K = None ∧ o = Psh_plain
      ∨ ∃ o_k : pshape,
          lookup (ap_exn_continuation ap) K
          = Some (Kshape [o_k] [Value] false)
          ∧ fits o o_k))
  ∨ ∃ (o_k : pshape) (sh : list pshape) (KS : list flambda_kind),
      lookup (ap_exn_continuation ap) K
      = Some (Kshape (o_k :: sh) (Value :: KS) true)
      ∧ fits o o_k
      ∧ args_rel2 sh (exn_extra_simples ap) (exn_extra_simples ap')
      ∧ length (ap_exn_extra_args ap) = length KS.

(* The continuations of the body of code. *)
Definition code_K (c : code) (cs : code_shape) : finite_map kshape :=
  [(code_return_continuation c,
    Kshape (cs_results cs) (cs_result_kinds cs) true);
   (code_exn_continuation c, Kshape [cs_exn cs] [Value] false)].

(* What code of the two runs share but for their bodies. *)
Definition code_frame (id : code_id) (c c' : code) (cs : code_shape)
    : Prop :=
  CS id = Some cs
  ∧ code_params_arity c' = cs_params_arity cs
  ∧ code_result_arity c' = cs_result_arity cs
  ∧ length (code_params_arity c) = length (cs_params_arity cs)
  ∧ (cs_keeps cs = true ->
     code_params_arity c = cs_params_arity cs
     ∧ code_result_arity c = cs_result_arity cs)
  ∧ code_is_tupled c = false ∧ code_is_tupled c' = false
  ∧ code_my_closure c' = code_my_closure c
  ∧ fields_of M (code_my_closure c) = None
  ∧ among (code_my_closure c) (field_variables M) = false
  ∧ code_return_continuation c' = code_return_continuation c
  ∧ code_exn_continuation c' = code_exn_continuation c
  ∧ binds_statics (code_body c') = binds_statics (code_body c)
  ∧ ∃ (sh : list pshape) (pk pk' : list (variable * flambda_kind)),
      code_params c = map fst pk ∧ code_params c' = map fst pk'
      ∧ params_rel2 sh pk pk' ∧ Forall2 fits (cs_params cs) sh.

Inductive uc : finite_map kshape -> expr -> expr -> Prop :=
  | uc_let (K : finite_map kshape) (x : variable) (kind : flambda_kind)
      (d : named) (body body' : expr) :
    plain2 x = true -> named_plain2 d = true ->
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    uc K body body' ->
    uc K (Let x kind d body) (Let x kind d body')
  | uc_alloc (K : finite_map kshape) (x : variable) (d : named)
      (fields : unboxed_fields)
      (ls : list (list (variable * flambda_kind * simple)))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    Forall2 (alloc_entry2 x d) fields ls ->
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    uc K body body' ->
    uc K (Let x Value d body) (lets (concat ls) body')
  | uc_alias (K : finite_map kshape) (x y : variable)
      (fields fields_y : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    fields_of M y = Some fields_y -> x <> y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs ->
    uc K body body' ->
    uc K (Let x Value (Simple (Var y)) body) (lets (pairs_lets pairs) body')
  | uc_load (K : finite_map kshape) (x : variable) (kind : flambda_kind)
      (d : named) (f : field) (s : simple) (y : variable)
      (fields fields_y nested : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields_y -> x <> y ->
    (f, Unboxed nested) ∈ fields_y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed nested)
    = Some pairs ->
    uc K body body' ->
    uc K (Let x Value d body) (lets (pairs_lets pairs) body')
  | uc_read (K : finite_map kshape) (x : variable) (kind : flambda_kind)
      (d : named) (f : field) (s : simple) (y v : variable)
      (fields : unboxed_fields) (body body' : expr) :
    plain2 x = true ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields ->
    (f, Not_unboxed v) ∈ fields ->
    uc K body body' ->
    uc K (Let x kind d body) (Let x kind (Simple (Var v)) body')
  | uc_static (K : finite_map kshape) (bindings bindings' : list static_binding)
      (body body' : expr) :
    uc_bindings bindings bindings' -> uc K body body' ->
    uc K (Let_static bindings body) (Let_static bindings' body')
  | uc_let_cont (K : finite_map kshape) (k : continuation) (sh : list pshape)
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata)
      (handler handler' body body' : expr) :
    params_rel2 sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> uc K handler handler' ->
    uc ((k, Kshape sh (map snd ps) true) :: K) body body' ->
    uc K
      (Let_cont_non_recursive k (Handler (map fst ps) metadata handler) body)
      (Let_cont_non_recursive k (Handler (map fst ps') metadata' handler')
         body')
  | uc_let_cont_rec (K S : finite_map kshape)
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    uc_handlers (S ++ K) handlers handlers' S ->
    handlers_bind_statics handlers = handlers_bind_statics handlers' ->
    uc (S ++ K) body body' ->
    uc K (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers' body')
  | uc_apply_cont (K : finite_map kshape) (ac ac' : apply_cont) :
    ac_rel2 K ac ac' -> uc K (Apply_cont ac) (Apply_cont ac')
  | uc_switch (K : finite_map kshape) (scrutinee : simple)
      (arms arms' : list (Z * apply_cont)) :
    plain_simple2 scrutinee = true ->
    Forall2
      (fun (arm arm' : Z * apply_cont) =>
         fst arm' = fst arm ∧ ac_rel2 K (snd arm) (snd arm'))
      arms arms' ->
    uc K (Switch scrutinee arms) (Switch scrutinee arms')
  | uc_invalid (K : finite_map kshape) : uc K Invalid Invalid
  | uc_call (K : finite_map kshape) (ap ap' : apply_expr) (cs : code_shape)
      (k : continuation) (sh : list pshape) :
    call_rel ap ap' cs -> exn_call_rel K (cs_exn cs) ap ap' ->
    ap_continuation ap = k -> ap_continuation ap' = k ->
    lookup k K = Some (Kshape sh (cs_result_kinds cs) true) ->
    transport (cs_results cs) sh (cs_result_kinds cs) ->
    uc K (Apply ap) (Apply ap')
  | uc_call_wrap (K : finite_map kshape) (ap ap' : apply_expr)
      (cs : code_shape) (k : continuation) (sh : list pshape)
      (w : continuation) (wps : list variable) (wmd : handler_metadata)
      (wargs : list simple) :
    call_rel ap ap' cs -> exn_call_rel K (cs_exn cs) ap ap' ->
    ap_continuation ap = k -> ap_continuation ap' = w ->
    w <> ap_exn_continuation ap ->
    lookup k K = Some (Kshape sh (cs_result_kinds cs) true) ->
    wrap_u_ok (cs_results cs) sh (cs_result_kinds cs) wps wmd wargs ->
    uc K (Apply ap) (wrap w wps wmd k wargs (Apply ap'))
  | uc_call_wraps (K : finite_map kshape) (ap ap' : apply_expr)
      (cs : code_shape) (k : continuation) (sh : list pshape)
      (KS : list flambda_kind) (wc : continuation) (cps : list variable)
      (cmd : handler_metadata) (cargs : list simple) (w : continuation)
      (wps : list variable) (wmd : handler_metadata) (wargs : list simple) :
    call_rel ap ap' cs -> exn_call_rel K (cs_exn cs) ap ap' ->
    ap_continuation ap = wc -> ap_continuation ap' = w ->
    wc <> ap_exn_continuation ap -> w <> ap_exn_continuation ap ->
    lookup k K = Some (Kshape sh KS true) ->
    hm_params_kinds cmd = cs_result_kinds cs ->
    wrap_both_ok (cs_results cs) sh cps cmd cargs wps wmd wargs ->
    uc K (wrap wc cps cmd k cargs (Apply ap))
      (wrap w wps wmd k wargs (Apply ap'))
with uc_handlers : finite_map kshape
    -> list (continuation * continuation_handler)
    -> list (continuation * continuation_handler)
    -> finite_map kshape -> Prop :=
  | uc_handlers_nil (K : finite_map kshape) : uc_handlers K [] [] []
  | uc_handlers_cons (K : finite_map kshape) (k : continuation)
      (sh : list pshape) (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (handlers handlers' : list (continuation * continuation_handler))
      (S : finite_map kshape) :
    params_rel2 sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> uc K handler handler' ->
    uc_handlers K handlers handlers' S ->
    uc_handlers K ((k, Handler (map fst ps) metadata handler) :: handlers)
      ((k, Handler (map fst ps') metadata' handler') :: handlers')
      ((k, Kshape sh (map snd ps) true) :: S)
with uc_bindings : list static_binding -> list static_binding -> Prop :=
  | uc_bindings_nil : uc_bindings [] []
  | uc_bindings_symbol (b : static_binding) (bs bs' : list static_binding) :
    local_binding2 b = true -> uc_bindings bs bs' ->
    uc_bindings (b :: bs) (b :: bs')
  | uc_bindings_deleted (id : code_id) (bs bs' : list static_binding) :
    uc_bindings bs bs' ->
    uc_bindings (Bind_deleted_code id :: bs) (Bind_deleted_code id :: bs')
  | uc_bindings_code (id : code_id) (c c' : code) (cs : code_shape)
      (bs bs' : list static_binding) :
    code_frame id c c' cs -> uc (code_K c cs) (code_body c) (code_body c') ->
    uc_bindings bs bs' ->
    uc_bindings (Bind_code id c :: bs) (Bind_code id c' :: bs').

Definition code_rel (id : code_id) (c c' : code) : Prop :=
  ∃ cs : code_shape,
    code_frame id c c' cs ∧ uc (code_K c cs) (code_body c) (code_body c').

Definition statics_rel (S S' : static_env) : Prop :=
  static_values S' = static_values S
  ∧ map fst (static_code S') = map fst (static_code S)
  ∧ ∀ (id : code_id) (c : code),
      lookup id (static_code S) = Some c ->
      ∃ c' : code, lookup id (static_code S') = Some c' ∧ code_rel id c c'.

Definition env_rel2 (vars vars' : env) : Prop :=
  (∀ z : variable, plain2 z = true -> lookup z vars = lookup z vars')
  ∧ ∀ (x : variable) (fields : unboxed_fields) (w : value),
      fields_of M x = Some fields -> lookup x vars = Some w ->
      fields_hold vars' fields w.

Inductive cont_rel : kshape -> cont_value -> cont_value -> Prop :=
  | cont_rel_closure (ksh : kshape) (sh0 : list pshape)
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (vars vars' : env) (K S : finite_map kshape) (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    params_rel2 sh0 ps ps' -> transport (ks_shapes ksh) sh0 (ks_kinds ksh) ->
    hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' ->
    ks_kinds ksh = map snd ps ->
    uc (S ++ K) handler handler' -> env_rel2 vars vars' ->
    conts_rel K conts conts' -> uc_handlers (S ++ K) recursive recursive' S ->
    cont_rel ksh
      (Closure (Handler (map fst ps) metadata handler) vars conts recursive)
      (Closure (Handler (map fst ps') metadata' handler') vars' conts'
         recursive')
  | cont_rel_outer (ksh : kshape) (k : continuation) :
    Forall (fun o => o = Psh_plain) (ks_shapes ksh) ->
    ks_handler ksh = false ->
    cont_rel ksh (Outer k) (Outer k)
  | cont_rel_wrap_u (sh_r sh_k : list pshape) (KS : list flambda_kind)
      (b : bool) (c : cont_value) (wps : list variable)
      (wmd : handler_metadata) (k : continuation) (wargs : list simple)
      (wvars : env) (wconts : cont_env) :
    cont_rel (Kshape sh_k KS true) c (resolve k wconts) ->
    wrap_u_ok sh_r sh_k KS wps wmd wargs ->
    cont_rel (Kshape sh_r KS b) c
      (wrapper_closure wps wmd k wargs wvars wconts)
  | cont_rel_wrap_both (sh_r sh_k : list pshape) (KS : list flambda_kind)
      (b : bool) (cps : list variable) (cmd : handler_metadata)
      (cargs : list simple) (cvars : env) (cconts : cont_env)
      (wps : list variable) (wmd : handler_metadata) (wargs : list simple)
      (wvars : env) (wconts : cont_env) (k : continuation) :
    cont_rel (Kshape sh_k KS true) (resolve k cconts) (resolve k wconts) ->
    wrap_both_ok sh_r sh_k cps cmd cargs wps wmd wargs ->
    cont_rel (Kshape sh_r (hm_params_kinds cmd) b)
      (wrapper_closure cps cmd k cargs cvars cconts)
      (wrapper_closure wps wmd k wargs wvars wconts)
  | cont_rel_extra (o o' : pshape) (sh : list pshape)
      (KS : list flambda_kind) (c c' : cont_value)
      (extra extra' : list value) :
    cont_rel (Kshape (o' :: sh) (Value :: KS) true) c c' -> fits o o' ->
    length extra = length KS -> vrel sh extra extra' ->
    cont_rel (Kshape [o] [Value] false) (with_extra_args c extra)
      (with_extra_args c' extra')
  | cont_rel_over (sh : list pshape) (KS : list flambda_kind) (b : bool)
      (cs : code_shape) (remaining : list value) (ra : arity)
      (c c' e e' : cont_value) :
    transport sh [Psh_plain] KS -> KS = [Value] ->
    cont_rel (Kshape (cs_results cs) (cs_result_kinds cs) true) c c' ->
    cont_rel (Kshape [cs_exn cs] [Value] false) e e' ->
    ind_targets cs ->
    cont_rel (Kshape sh KS b)
      (Over_application remaining ra (cs_result_arity cs) c e)
      (Over_application remaining ra (cs_result_arity cs) c' e')
with conts_rel : finite_map kshape -> cont_env -> cont_env -> Prop :=
  | conts_rel_nil : conts_rel [] [] []
  | conts_rel_cons (k : continuation) (ksh : kshape) (c c' : cont_value)
      (K : finite_map kshape) (conts conts' : cont_env) :
    cont_rel ksh c c' -> conts_rel K conts conts' ->
    conts_rel ((k, ksh) :: K) ((k, c) :: conts) ((k, c') :: conts').

(* The states related: by the terms; both at a call, its continuations
   related; or both applying a continuation to arguments related. *)
Definition related_main (s t : state) : Prop :=
  ∃ K : finite_map kshape,
    uc K (state_expr s) (state_expr t)
    ∧ env_rel2 (state_vars s) (state_vars t)
    ∧ conts_rel K (state_conts s) (state_conts t)
    ∧ statics_rel (state_statics s) (state_statics t)
    ∧ state_heap s = empty_heap ∧ state_heap t = empty_heap
    ∧ state_reads s = state_reads t.

Definition related_call (s t : state) : Prop :=
  ∃ (ap ap' : apply_expr) (cs : code_shape),
    state_expr s = Apply ap ∧ state_expr t = Apply ap'
    ∧ call_rel ap ap' cs
    ∧ env_rel2 (state_vars s) (state_vars t)
    ∧ cont_rel (Kshape (cs_results cs) (cs_result_kinds cs) true)
        (resolve (ap_continuation ap) (state_conts s))
        (resolve (ap_continuation ap') (state_conts t))
    ∧ (∀ extra : list value,
         eval_simples (state_vars s) (exn_extra_simples ap) = Some extra ->
         ∃ extra' : list value,
           eval_simples (state_vars t) (exn_extra_simples ap') = Some extra'
           ∧ cont_rel (Kshape [cs_exn cs] [Value] false)
               (with_extra_args
                  (resolve (ap_exn_continuation ap) (state_conts s)) extra)
               (with_extra_args
                  (resolve (ap_exn_continuation ap') (state_conts t))
                  extra'))
    ∧ statics_rel (state_statics s) (state_statics t)
    ∧ state_heap s = empty_heap ∧ state_heap t = empty_heap
    ∧ state_reads s = state_reads t.

Definition related_pending (s t : state) : Prop :=
  ∃ (ac ac' : apply_cont) (ksh : kshape),
    state_expr s = Apply_cont ac ∧ state_expr t = Apply_cont ac'
    ∧ ac_continuation ac' = ac_continuation ac
    ∧ cont_rel ksh (resolve (ac_continuation ac) (state_conts s))
        (resolve (ac_continuation ac') (state_conts t))
    ∧ (∀ vs : list value,
         eval_simples (state_vars s) (ac_args ac) = Some vs ->
         ∃ vs' : list value,
           eval_simples (state_vars t) (ac_args ac') = Some vs'
           ∧ vrel (ks_shapes ksh) vs vs')
    ∧ statics_rel (state_statics s) (state_statics t)
    ∧ state_heap s = empty_heap ∧ state_heap t = empty_heap
    ∧ state_reads s = state_reads t.

Definition related2 (s t : state) : Prop :=
  related_main s t ∨ related_call s t ∨ related_pending s t.

End Related.

(* ======================================================================== *)
(* Paths and values                                                         *)
(* ======================================================================== *)

Lemma forall2_map_r {A B C : Type} (P : A -> C -> Prop) (g : B -> C)
    (l : list A) (l' : list B) :
  Forall2 (fun a b => P a (g b)) l l' -> Forall2 P l (map g l').
Proof. induction 1; constructor; assumption. Qed.

Lemma forall2_impl {A B : Type} (P Q : A -> B -> Prop) (l : list A)
    (l' : list B) :
  Forall2 P l l' -> (∀ a b, P a b -> Q a b) -> Forall2 Q l l'.
Proof. intros HP H. induction HP; constructor; auto. Qed.

Lemma ufield_paths_cons (k : flambda_kind) (f : field) (u : unboxed_field)
    (rest : unboxed_fields) :
  ufield_paths k (Unboxed ((f, u) :: rest))
  = map (fun pk => (f :: fst pk, snd pk)) (ufield_paths (field_kind f) u)
    ++ ufield_paths k (Unboxed rest).
Proof. reflexivity. Qed.

Lemma ufield_leaves_cons (k : flambda_kind) (f : field) (u : unboxed_field)
    (rest : unboxed_fields) :
  unboxed_field_leaves k (Unboxed ((f, u) :: rest))
  = unboxed_field_leaves (field_kind f) u
    ++ unboxed_field_leaves k (Unboxed rest).
Proof. reflexivity. Qed.

(* Each leaf at its path. *)
Lemma leaves_paths (u : unboxed_field) :
  ∀ k : flambda_kind,
    Forall2
      (fun (vk : variable * flambda_kind) (pk : list field * flambda_kind) =>
         snd vk = snd pk ∧ upath k u (fst pk) (fst vk) (snd vk))
      (unboxed_field_leaves k u) (ufield_paths k u).
Proof.
  induction u as [v | fields IH] using unboxed_field_rec; intros k.
  - simpl. constructor; [ split; [ reflexivity | constructor ] | constructor ].
  - induction fields as [ | [f0 u0] rest IHrest]; [ constructor | ].
    rewrite ufield_leaves_cons, ufield_paths_cons.
    apply Forall2_app.
    + apply forall2_map_r.
      apply (forall2_impl _ _ _ _
               (IH f0 u0 (or_introl eq_refl) (field_kind f0))).
      intros [v kv] [p kp] [Hk Hp]. simpl in *.
      split; [ exact Hk | ]. econstructor; [ left; reflexivity | exact Hp ].
    + apply (forall2_impl _ _ _ _
               (IHrest (fun f u Hin => IH f u (or_intror Hin)))).
      intros [v kv] [p kp] [Hk Hp]. simpl in *.
      split; [ exact Hk | ]. apply upath_cons. exact Hp.
Qed.

Lemma fields_leaves_paths (fields : unboxed_fields) :
  Forall2
    (fun (vk : variable * flambda_kind) (pk : list field * flambda_kind) =>
       snd vk = snd pk
       ∧ upath Value (Unboxed fields) (fst pk) (fst vk) (snd vk))
    (fields_leaves fields) (fields_paths fields).
Proof. exact (leaves_paths (Unboxed fields) Value). Qed.

(* The second variable of each pair is at the path of the first. *)
Lemma pairs_paths (u : unboxed_field) :
  ∀ (k : flambda_kind) (u' : unboxed_field)
    (pairs : list (variable * variable * flambda_kind)),
    unboxed_field_pairs k u u' = Some pairs ->
    Forall2
      (fun (xyk : variable * variable * flambda_kind)
           (pk : list field * flambda_kind) =>
         let '(_, x', k1) := xyk in k1 = snd pk ∧ upath k u' (fst pk) x' k1)
      pairs (ufield_paths k u).
Proof.
  induction u as [v | fields IH] using unboxed_field_rec;
    intros k u' pairs Hp.
  - destruct u' as [v' | fields']; [ | discriminate Hp ].
    injection Hp as <-. simpl. constructor; [ | constructor ].
    split; [ reflexivity | constructor ].
  - destruct u' as [v' | fields']; [ discriminate Hp | ].
    revert pairs Hp. induction fields as [ | [f0 u0] rest IHrest];
      intros pairs Hp.
    + injection Hp as <-. constructor.
    + change (unboxed_field_pairs k (Unboxed ((f0, u0) :: rest))
                (Unboxed fields'))
        with (match Keyed.lookup field_eqb f0 fields' with
              | Some u' =>
                match unboxed_field_pairs (field_kind f0) u0 u',
                      unboxed_field_pairs k (Unboxed rest) (Unboxed fields')
                with
                | Some p, Some q => Some (p ++ q)
                | _, _ => None
                end
              | None => None
              end) in Hp.
      destruct (Keyed.lookup field_eqb f0 fields') as [u0' | ] eqn:Hl;
        [ | discriminate Hp ].
      pose proof (keyed_lookup_in field_eqb field_eqb_decides f0 _ fields'
                    Hl) as Hin0.
      destruct (unboxed_field_pairs (field_kind f0) u0 u0') as [p1 | ]
        eqn:Hp1; [ | discriminate Hp ].
      destruct (unboxed_field_pairs k (Unboxed rest) (Unboxed fields'))
        as [q | ] eqn:Hq; [ | discriminate Hp ].
      injection Hp as <-. rewrite ufield_paths_cons.
      apply Forall2_app.
      * apply forall2_map_r.
        apply (forall2_impl _ _ _ _
                 (IH f0 u0 (or_introl eq_refl) (field_kind f0) u0' p1 Hp1)).
        intros [[x x'] k1] [p kp] [Hk Hpath]. simpl in *.
        split; [ exact Hk | ]. econstructor; [ exact Hin0 | exact Hpath ].
      * exact (IHrest (fun f u Hin => IH f u (or_intror Hin)) q eq_refl).
Qed.

Lemma ufield_paths_length (u : unboxed_field) :
  ∀ k : flambda_kind,
    length (ufield_paths k u) = length (unboxed_field_leaves k u).
Proof.
  intros k. symmetry. exact (Forall2_length (leaves_paths u k)).
Qed.

(* The leaf at a path has an index, the same in the leaves and the
   paths. *)
Lemma upath_index (k : flambda_kind) (u : unboxed_field) (p : list field)
    (v : variable) (k' : flambda_kind) :
  upath k u p v k' ->
  ∃ i : nat,
    nth_error (unboxed_field_leaves k u) i = Some (v, k')
    ∧ nth_error (ufield_paths k u) i = Some (p, k').
Proof.
  induction 1 as [k v | k fields f u p v k' Hin _ IH].
  - exists 0. split; reflexivity.
  - destruct IH as (i & Hl & Hp).
    induction fields as [ | [f0 u0] rest IHrest]; [ destruct Hin | ].
    rewrite ufield_leaves_cons, ufield_paths_cons.
    destruct Hin as [Heq | Hin].
    + injection Heq as -> ->. exists i.
      assert (Hlt : i < length (unboxed_field_leaves (field_kind f) u))
        by (apply nth_error_Some; rewrite Hl; discriminate).
      split.
      * rewrite nth_error_app1 by exact Hlt. exact Hl.
      * rewrite nth_error_app1
          by (rewrite length_map, ufield_paths_length; exact Hlt).
        rewrite nth_error_map, Hp. reflexivity.
    + destruct (IHrest Hin) as (j & Hlj & Hpj).
      exists (length (unboxed_field_leaves (field_kind f0) u0) + j).
      split.
      * rewrite nth_error_app2 by lia.
        replace (length (unboxed_field_leaves (field_kind f0) u0) + j
                 - length (unboxed_field_leaves (field_kind f0) u0))
          with j by lia.
        exact Hlj.
      * rewrite nth_error_app2
          by (rewrite length_map, ufield_paths_length; lia).
        rewrite length_map, ufield_paths_length.
        replace (length (unboxed_field_leaves (field_kind f0) u0) + j
                 - length (unboxed_field_leaves (field_kind f0) u0))
          with j by lia.
        exact Hpj.
Qed.

Lemma fits_refl (o : pshape) : fits o o.
Proof. destruct o; constructor; reflexivity. Qed.

Lemma fits_list_refl (sh : list pshape) : Forall2 fits sh sh.
Proof. induction sh; constructor; [ apply fits_refl | assumption ]. Qed.

Lemma fits_trans (a b c : pshape) : fits a b -> fits b c -> fits a c.
Proof.
  intros Hab Hbc. inversion Hab; subst; inversion Hbc; subst;
    try constructor; congruence.
Qed.

Lemma fits_list_trans (a b c : list pshape) :
  Forall2 fits a b -> Forall2 fits b c -> Forall2 fits a c.
Proof.
  intros Hab. revert c. induction Hab as [ | x y a b Hxy _ IH];
    intros c Hbc; inversion Hbc; subst; constructor;
    [ exact (fits_trans _ _ _ Hxy ltac:(eassumption)) | auto ].
Qed.

Lemma vrel_fits (sh : list pshape) (vs vs' : list value) :
  vrel sh vs vs' -> ∀ sh' : list pshape, Forall2 fits sh sh' ->
  vrel sh' vs vs'.
Proof.
  induction 1 as [vs | o sh | sh v vs vs' _ IH
                 | sh kind v v' vs vs' Hv _ IH
                 | sh fields v ws vs vs' Hws _ IH];
    intros sh' Hf; inversion Hf as [ | a b l l' Hab Hl]; subst.
  - constructor.
  - constructor.
  - inversion Hab; subst; constructor; auto.
  - inversion Hab; subst. constructor; auto.
  - inversion Hab as [ | | | f1 f2 Hpaths]; subst. constructor; [ | auto ].
    unfold fields_vals in *. rewrite <- Hpaths. exact Hws.
Qed.

Lemma transport_refl (sh : list pshape) (KS : list flambda_kind) :
  transport sh sh KS.
Proof. intros vs vs' H _. exact H. Qed.

Lemma transport_fits (sh sh' : list pshape) (KS : list flambda_kind) :
  Forall2 fits sh sh' -> transport sh sh' KS.
Proof. intros Hf vs vs' H _. exact (vrel_fits _ _ _ H _ Hf). Qed.

Lemma transport_trans (a b c : list pshape) (KS : list flambda_kind) :
  transport a b KS -> transport b c KS -> transport a c KS.
Proof. intros Hab Hbc vs vs' H Hk. exact (Hbc _ _ (Hab _ _ H Hk) Hk). Qed.

Lemma vrel_plain_eq (sh : list pshape) (vs vs' : list value) :
  vrel sh vs vs' -> Forall (fun o => o = Psh_plain) sh -> vs' = vs.
Proof.
  induction 1 as [vs | o sh | sh v vs vs' _ IH | sh kind v v' vs vs' _ _ IH
                 | sh fields v ws vs vs' _ _ IH];
    intros Hall; try reflexivity.
  - inversion Hall; subst. rewrite IH; [ reflexivity | assumption ].
  - inversion Hall; subst. discriminate.
  - inversion Hall; subst. discriminate.
Qed.

Lemma fields_vals_kinds (fields : unboxed_fields) (v : value)
    (ws : list value) :
  fields_vals fields v ws ->
  map kind_of_value ws = map snd (fields_paths fields).
Proof.
  unfold fields_vals. induction 1 as [ | pk w l l' [_ Hk] _ IH];
    [ reflexivity | ].
  simpl. rewrite Hk, IH. reflexivity.
Qed.

Lemma leaves_kinds (fields : unboxed_fields) :
  map snd (fields_leaves fields) = map snd (fields_paths fields).
Proof.
  induction (fields_leaves_paths fields) as [ | vk pk l l' [Hk _] _ IH];
    [ reflexivity | ].
  simpl. rewrite Hk, IH. reflexivity.
Qed.

Lemma kind_of_poison (kind : flambda_kind) :
  kind_of_value (Value_const (Poison kind)) = kind.
Proof. reflexivity. Qed.

(* The kinds of the arguments of a call in the second run. *)
Lemma vrel_shape_kinds (sh : list pshape) (vs vs' : list value) :
  vrel sh vs vs' -> no_dead sh ->
  map kind_of_value vs' = shape_kinds sh (map kind_of_value vs).
Proof.
  induction 1 as [vs | o sh | sh v vs vs' _ IH | sh kind v v' vs vs' _ _ IH
                 | sh fields v ws vs vs' Hws _ IH];
    intros Hnd.
  - destruct vs; reflexivity.
  - destruct o; reflexivity.
  - simpl. rewrite IH; [ reflexivity | ].
    intros kind Hin. exact (Hnd kind (or_intror Hin)).
  - exfalso. exact (Hnd kind (or_introl eq_refl)).
  - simpl. rewrite map_app, (fields_vals_kinds fields v ws Hws), IH;
      [ reflexivity | ].
    intros kind Hin. exact (Hnd kind (or_intror Hin)).
Qed.

(* ======================================================================== *)
(* Binding                                                                  *)
(* ======================================================================== *)

Lemma bind_params_other (xs : list variable) :
  ∀ (ws : list value) (cv hv : env) (z : variable),
    bind_params xs ws cv = Some hv -> z ∉ xs -> lookup z hv = lookup z cv.
Proof.
  induction xs as [ | x xs IH]; intros [ | w ws] cv hv z Hb Hz;
    simpl in Hb; try discriminate Hb.
  - injection Hb as <-. reflexivity.
  - rewrite (IH ws _ hv z Hb (fun H => Hz (or_intror H))).
    apply lookup_other. intros ->. apply Hz. left. reflexivity.
Qed.

Lemma bind_params_nth (xs : list variable) :
  ∀ (ws : list value) (cv hv : env) (i : nat) (x : variable) (w : value),
    NoDup xs -> bind_params xs ws cv = Some hv ->
    nth_error xs i = Some x -> nth_error ws i = Some w ->
    lookup x hv = Some w.
Proof.
  induction xs as [ | x0 xs IH]; intros [ | w0 ws] cv hv i x w Hnd Hb Hx Hw;
    simpl in Hb; try discriminate Hb.
  - destruct i; discriminate Hx.
  - inversion Hnd as [ | x1 l Hx0 Hnd' [Heq Heq']]; subst.
    destruct i as [ | i].
    + simpl in Hx, Hw. injection Hx as <-. injection Hw as <-.
      rewrite (bind_params_other xs ws _ hv x0 Hb Hx0). apply lookup_head.
    + exact (IH ws _ hv i x w Hnd' Hb Hx Hw).
Qed.

Lemma bind_params_some (xs : list variable) :
  ∀ (ws : list value) (cv : env),
    length xs = length ws -> ∃ hv : env, bind_params xs ws cv = Some hv.
Proof.
  induction xs as [ | x xs IH]; intros [ | w ws] cv Hl; simpl in Hl;
    try discriminate Hl.
  - exists cv. reflexivity.
  - simpl. apply IH. lia.
Qed.

Lemma bind_params_length (xs : list variable) :
  ∀ (ws : list value) (cv hv : env),
    bind_params xs ws cv = Some hv -> length xs = length ws.
Proof.
  induction xs as [ | x xs IH]; intros [ | w ws] cv hv Hb; simpl in Hb;
    try discriminate Hb; [ reflexivity | ].
  simpl. f_equal. exact (IH ws _ hv Hb).
Qed.

Lemma forall2_nth {A B : Type} (P : A -> B -> Prop) (l : list A)
    (l' : list B) (i : nat) (a : A) (b : B) :
  Forall2 P l l' -> nth_error l i = Some a -> nth_error l' i = Some b ->
  P a b.
Proof.
  intros H. revert i. induction H as [ | a0 b0 l l' Hab _ IH];
    intros [ | i] Ha Hb; try discriminate Ha; simpl in Ha, Hb.
  - injection Ha as <-. injection Hb as <-. exact Hab.
  - exact (IH i Ha Hb).
Qed.

Lemma forall2_nth_l {A B : Type} (P : A -> B -> Prop) (l : list A)
    (l' : list B) (i : nat) (a : A) :
  Forall2 P l l' -> nth_error l i = Some a ->
  ∃ b : B, nth_error l' i = Some b ∧ P a b.
Proof.
  intros H. revert i. induction H as [ | a0 b0 l l' Hab _ IH];
    intros [ | i] Ha; try discriminate Ha; simpl in Ha.
  - injection Ha as <-. exists b0. split; [ reflexivity | exact Hab ].
  - exact (IH i Ha).
Qed.

(* ======================================================================== *)
(* Related environments                                                     *)
(* ======================================================================== *)

Section Envs.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.
Variable dead : list variable.
Variable CS : code_id -> option code_shape.

Lemma plain2_iff (z : variable) :
  plain2 M dead z = true
  <-> plain_variable M z = true ∧ among z dead = false.
Proof.
  unfold plain2. rewrite andb_true_iff, negb_true_iff. reflexivity.
Qed.

Lemma plain_simple2_eval (vars vars' : env) (s : simple) :
  env_rel2 M dead vars vars' -> plain_simple2 M dead s = true ->
  eval_simple vars s = eval_simple vars' s.
Proof.
  intros [Hplain _] Hs. destruct s as [x | sym | c]; simpl; try reflexivity.
  exact (Hplain x Hs).
Qed.

Lemma plain_simples2_eval (vars vars' : env) (args : list simple) :
  env_rel2 M dead vars vars' -> forallb (plain_simple2 M dead) args = true ->
  eval_simples vars args = eval_simples vars' args.
Proof.
  intros Hrel. induction args as [ | s args IH]; simpl; [ reflexivity | ].
  intros Hargs. apply andb_true_iff in Hargs as [Hs Hargs].
  rewrite (plain_simple2_eval vars vars' s Hrel Hs), (IH Hargs).
  reflexivity.
Qed.

Lemma env_rel2_bind (vars vars' : env) (x : variable) (w : value) :
  env_rel2 M dead vars vars' -> plain2 M dead x = true ->
  env_rel2 M dead ((x, w) :: vars) ((x, w) :: vars').
Proof.
  intros [Hplain Hfields] Hx. apply plain2_iff in Hx as [Hx _]. split.
  - intros z Hz. rewrite !lookup_cons. destruct (Pos.eqb z x);
      [ reflexivity | exact (Hplain z Hz) ].
  - intros y fields w0 Hy Hlookup p v k Hp.
    rewrite lookup_other in Hlookup
      by (intros ->; exact (plain_not_unboxed M x fields Hx Hy)).
    destruct (Hfields y fields w0 Hy Hlookup p v k Hp) as (w' & Hw' & Hv & Hk).
    exists w'. split; [ exact Hw' | split; [ | exact Hk ] ].
    rewrite lookup_other; [ exact Hv | ]. intros ->.
    apply plain_variable_iff in Hx as [_ Hx].
    pose proof (field_variables_in M y fields x Hy
                  (upath_leaf_in fields p x k Hp)) as Hfv.
    apply among_In in Hfv. rewrite Hfv in Hx. discriminate.
Qed.

Lemma env_rel2_bind_dead (vars vars' : env) (x : variable) (w w' : value) :
  env_rel2 M dead vars vars' -> among x dead = true ->
  fields_of M x = None -> among x (field_variables M) = false ->
  env_rel2 M dead ((x, w) :: vars) ((x, w') :: vars').
Proof.
  intros [Hplain Hfields] Hd Hx Hxf. split.
  - intros z Hz. apply plain2_iff in Hz as Hz'. destruct Hz' as [_ Hzd].
    rewrite !lookup_other by (intros ->; rewrite Hd in Hzd; discriminate).
    exact (Hplain z Hz).
  - intros y fields w0 Hy Hlookup p v k Hp.
    rewrite lookup_other in Hlookup by (intros ->; congruence).
    destruct (Hfields y fields w0 Hy Hlookup p v k Hp) as (w1 & Hw1 & Hv & Hk).
    exists w1. split; [ exact Hw1 | split; [ | exact Hk ] ].
    rewrite lookup_other; [ exact Hv | ]. intros ->.
    pose proof (field_variables_in M y fields x Hy
                  (upath_leaf_in fields p x k Hp)) as Hfv.
    apply among_In in Hfv. rewrite Hfv in Hxf. discriminate.
Qed.

Lemma env_rel2_bind_same (vars vars' : env) (x : variable) (w : value) :
  env_rel2 M dead vars vars' -> fields_of M x = None ->
  among x (field_variables M) = false ->
  env_rel2 M dead ((x, w) :: vars) ((x, w) :: vars').
Proof.
  intros [Hplain Hfields] Hx Hxf. split.
  - intros z Hz. rewrite !lookup_cons. destruct (Pos.eqb z x);
      [ reflexivity | exact (Hplain z Hz) ].
  - intros y fields w0 Hy Hlookup p v k Hp.
    rewrite lookup_other in Hlookup by (intros ->; congruence).
    destruct (Hfields y fields w0 Hy Hlookup p v k Hp) as (w1 & Hw1 & Hv & Hk).
    exists w1. split; [ exact Hw1 | split; [ | exact Hk ] ].
    rewrite lookup_other; [ exact Hv | ]. intros ->.
    pose proof (field_variables_in M y fields x Hy
                  (upath_leaf_in fields p x k Hp)) as Hfv.
    apply among_In in Hfv. rewrite Hfv in Hxf. discriminate.
Qed.

Lemma env_rel2_bind_fields (vars vars' vars1 : env) (x : variable)
    (fields : unboxed_fields) (w : value) :
  env_rel2 M dead vars vars' -> fields_of M x = Some fields ->
  (∀ z : variable,
     z ∉ leaf_variables fields -> lookup z vars1 = lookup z vars') ->
  fields_hold vars1 fields w ->
  env_rel2 M dead ((x, w) :: vars) vars1.
Proof.
  intros [Hplain Hfs] Hx Hother Hhold. split.
  - intros z Hz. apply plain2_iff in Hz as Hz'. destruct Hz' as [Hzp _].
    rewrite lookup_other
      by (intros ->; exact (plain_not_unboxed M x fields Hzp Hx)).
    rewrite (Hplain z Hz), Hother; [ reflexivity | ].
    intros Hin. apply plain_variable_iff in Hzp as [_ Hzp].
    pose proof (field_variables_in M x fields z Hx Hin) as Hfv.
    apply among_In in Hfv. rewrite Hfv in Hzp. discriminate.
  - intros y fy w0 Hy Hlookup p v k Hp.
    destruct (Pos.eq_dec y x) as [-> | Hne].
    + rewrite lookup_head in Hlookup. injection Hlookup as <-.
      rewrite Hx in Hy. injection Hy as <-. exact (Hhold p v k Hp).
    + rewrite lookup_other in Hlookup by exact Hne.
      destruct (Hfs y fy w0 Hy Hlookup p v k Hp) as (w' & Hpw & Hl & Hk).
      exists w'. split; [ exact Hpw | split; [ | exact Hk ] ].
      rewrite Hother; [ exact Hl | ]. intros Hvx.
      exact (leaves_apart M M_distinct y x fy fields v Hy Hx Hne
               (upath_leaf_in fy p v k Hp) Hvx).
Qed.

Lemma env_rel2_empty : env_rel2 M dead [] [].
Proof.
  split; [ reflexivity | ]. intros x fields w _ Hlookup.
  discriminate Hlookup.
Qed.

(* The variables of the fields of a parameter bound to the values of its
   leaves. *)
Lemma bind_fields_hold (fields : unboxed_fields) (p : variable)
    (v : value) (ws : list value) (cv cv1 : env) :
  fields_of M p = Some fields -> fields_vals fields v ws ->
  bind_params (leaf_variables fields) ws cv = Some cv1 ->
  fields_hold cv1 fields v.
Proof.
  intros Hp Hws Hb path var k Hpath.
  destruct (upath_index _ _ _ _ _ Hpath) as (i & Hl & Hpi).
  destruct (forall2_nth_l _ _ _ i _ Hws Hpi) as (w' & Hw' & Hpw & Hk).
  exists w'. split; [ exact Hpw | split; [ | exact Hk ] ].
  apply (bind_params_nth _ ws cv cv1 i var w'
           (leaves_nodup M M_distinct p fields Hp) Hb); [ | exact Hw' ].
  unfold leaf_variables. rewrite nth_error_map.
  change (unboxed_field_leaves Value (Unboxed fields))
    with (fields_leaves fields) in Hl.
  rewrite Hl. reflexivity.
Qed.

(* Values related, given to parameters related, bind them so. *)
Lemma bind_vrel (sh : list pshape) (vs vs' : list value) :
  vrel sh vs vs' ->
  ∀ (ps ps' : list (variable * flambda_kind)) (cv cv' hv : env),
    params_rel2 M dead sh ps ps' -> env_rel2 M dead cv cv' ->
    bind_params (map fst ps) vs cv = Some hv ->
    ∃ hv' : env,
      bind_params (map fst ps') vs' cv' = Some hv' ∧ env_rel2 M dead hv hv'.
Proof.
  induction 1 as [vs | o sh | sh v vs vs' _ IH | sh kind v v' vs vs' _ _ IH
                 | sh fields v ws vs vs' Hws _ IH];
    intros ps ps' cv cv' hv Hps Hcv Hb.
  - inversion Hps; subst. destruct vs; simpl in Hb; [ | discriminate Hb ].
    injection Hb as <-. exists cv'. split; [ reflexivity | exact Hcv ].
  - inversion Hps; subst; discriminate Hb.
  - inversion Hps as [ | sh0 p kind ps1 ps1' Hp Hps1 | | ]; subst.
    simpl in Hb |- *.
    exact (IH ps1 ps1' _ _ hv Hps1 (env_rel2_bind cv cv' p v Hcv Hp) Hb).
  - inversion Hps as [ | | sh0 p kind0 ps1 ps1' Hd Hpf Hpv Hps1 | ]; subst.
    simpl in Hb |- *.
    exact (IH ps1 ps1' _ _ hv Hps1
             (env_rel2_bind_dead cv cv' p v v' Hcv Hd Hpf Hpv) Hb).
  - inversion Hps as [ | | | sh0 p fields0 ps1 ps1' Hp Hpv Hps1]; subst.
    simpl in Hb.
    assert (Hlen : length (leaf_variables fields) = length ws).
    { unfold leaf_variables, fields_vals, fields_paths, fields_leaves in *.
      rewrite length_map, <- (Forall2_length Hws), ufield_paths_length.
      reflexivity. }
    destruct (bind_params_some _ ws cv' Hlen) as (cv1 & Hcv1).
    assert (Hrel1 : env_rel2 M dead ((p, v) :: cv) cv1).
    { apply (env_rel2_bind_fields cv cv' cv1 p fields v Hcv Hp).
      - intros z Hz. exact (bind_params_other _ ws cv' cv1 z Hcv1 Hz).
      - exact (bind_fields_hold fields p v ws cv' cv1 Hp Hws Hcv1). }
    destruct (IH ps1 ps1' _ _ hv Hps1 Hrel1 Hb) as (hv' & Hb' & Hrel).
    exists hv'. split; [ | exact Hrel ].
    rewrite map_app. change (map fst (fields_leaves fields))
      with (leaf_variables fields).
    rewrite (bind_params_app _ _ _ _ _ _ Hcv1). exact Hb'.
Qed.

(* Values related at the parameters of a handler are of their kinds in
   the second run if they are in the first. *)
Lemma vrel_kinds_params (sh : list pshape) (vs vs' : list value) :
  vrel sh vs vs' ->
  ∀ ps ps' : list (variable * flambda_kind),
    params_rel2 M dead sh ps ps' ->
    map kind_of_value vs = map snd ps ->
    map kind_of_value vs' = map snd ps'.
Proof.
  induction 1 as [vs | o sh | sh v vs vs' _ IH | sh kind v v' vs vs' Hv _ IH
                 | sh fields v ws vs vs' Hws _ IH];
    intros ps ps' Hps Hk.
  - inversion Hps; subst. exact Hk.
  - inversion Hps; subst; discriminate Hk.
  - inversion Hps; subst. simpl in Hk |- *. injection Hk as Hk1 Hk.
    rewrite Hk1, (IH _ _ ltac:(eassumption) Hk). reflexivity.
  - inversion Hps; subst. simpl in Hk |- *. injection Hk as Hk1 Hk.
    rewrite (IH _ _ ltac:(eassumption) Hk).
    destruct Hv as [-> | Hv]; [ rewrite Hk1 | rewrite Hv ]; reflexivity.
  - inversion Hps; subst. simpl in Hk |- *. injection Hk as _ Hk.
    rewrite !map_app, (IH _ _ ltac:(eassumption) Hk),
      (fields_vals_kinds fields v ws Hws), leaves_kinds.
    reflexivity.
Qed.

(* The variables of the fields of an argument hold the values of the
   leaves of the parameter's fields. *)
Lemma pairs_eval (vars' : env) (fields_y : unboxed_fields) (w : value)
    (pairs : list (variable * variable * flambda_kind))
    (paths : list (list field * flambda_kind)) :
  Forall2
    (fun (xyk : variable * variable * flambda_kind)
         (pk : list field * flambda_kind) =>
       let '(_, x', k1) := xyk in
       k1 = snd pk ∧ upath Value (Unboxed fields_y) (fst pk) x' k1)
    pairs paths ->
  fields_hold vars' fields_y w ->
  ∃ ws : list value,
    eval_simples vars' (map (fun '(_, x', _) => Var x') pairs) = Some ws
    ∧ Forall2
        (fun (pk : list field * flambda_kind) (w' : value) =>
           path_value (fst pk) w = Some w' ∧ kind_of_value w' = snd pk)
        paths ws.
Proof.
  intros H Hhold. induction H as [ | [[x x'] k1] pk pairs paths [Hk Hp] _ IH].
  - exists []. split; [ reflexivity | constructor ].
  - destruct IH as (ws & Hws & Hall).
    destruct (Hhold _ _ _ Hp) as (w' & Hpw & Hl & Hkw).
    exists (w' :: ws). split.
    + cbn [map eval_simples eval_simple]. rewrite Hl, Hws. reflexivity.
    + constructor; [ | exact Hall ]. split; [ exact Hpw | congruence ].
Qed.

(* Arguments related, in environments related, have values related. *)
Lemma args_vrel (sh : list pshape) (args args' : list simple) :
  args_rel2 M dead sh args args' ->
  ∀ (vars vars' : env) (vs : list value),
    env_rel2 M dead vars vars' -> eval_simples vars args = Some vs ->
    ∃ vs' : list value,
      eval_simples vars' args' = Some vs' ∧ vrel sh vs vs'.
Proof.
  induction 1 as [args Hargs | o sh | sh a args args' Ha _ IH
                 | sh kind a a' args args' Ha _ IH
                 | sh fields fields_y y pairs args args' Hy Hpairs _ IH];
    intros vars vars' vs Hvars Heval.
  - exists vs. split; [ | constructor ].
    rewrite <- (plain_simples2_eval vars vars' args Hvars Hargs). exact Heval.
  - simpl in Heval. injection Heval as <-. exists [].
    split; [ reflexivity | constructor ].
  - simpl in Heval.
    destruct (eval_simple vars a) as [w | ] eqn:Hw; [ | discriminate Heval ].
    destruct (eval_simples vars args) as [vs1 | ] eqn:Hvs1;
      [ | discriminate Heval ].
    injection Heval as <-.
    destruct (IH vars vars' vs1 Hvars Hvs1) as (vs' & Hvs' & Hrel).
    exists (w :: vs'). split; [ | constructor; exact Hrel ].
    simpl. rewrite <- (plain_simple2_eval vars vars' a Hvars Ha), Hw, Hvs'.
    reflexivity.
  - simpl in Heval.
    destruct (eval_simple vars a) as [w | ] eqn:Hw; [ | discriminate Heval ].
    destruct (eval_simples vars args) as [vs1 | ] eqn:Hvs1;
      [ | discriminate Heval ].
    injection Heval as <-.
    destruct (IH vars vars' vs1 Hvars Hvs1) as (vs' & Hvs' & Hrel).
    destruct Ha as [-> | [-> Ha]].
    + exists (Value_const (Poison kind) :: vs'). split.
      * simpl. rewrite Hvs'. reflexivity.
      * constructor; [ right; reflexivity | exact Hrel ].
    + exists (w :: vs'). split.
      * simpl. rewrite <- (plain_simple2_eval vars vars' a Hvars Ha), Hw, Hvs'.
        reflexivity.
      * constructor; [ left; reflexivity | exact Hrel ].
  - simpl in Heval.
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | discriminate Heval ].
    destruct (eval_simples vars args) as [vs1 | ] eqn:Hvs1;
      [ | discriminate Heval ].
    injection Heval as <-.
    destruct (IH vars vars' vs1 Hvars Hvs1) as (vs' & Hvs' & Hrel).
    destruct (pairs_eval vars' fields_y w pairs (fields_paths fields)
                (pairs_paths (Unboxed fields) Value (Unboxed fields_y) pairs
                   Hpairs)
                (proj2 Hvars y fields_y w Hy Hw))
      as (ws & Hws & Hall).
    exists (ws ++ vs'). split.
    + rewrite FlambdaSemanticsProofs.eval_simples_app, Hws, Hvs'.
      reflexivity.
    + constructor; [ exact Hall | exact Hrel ].
Qed.

Lemma named_plain2_exec (statics statics' : static_env) (hp : heap)
    (rcd : read_record) (vars vars' : env) (d : named) :
  static_values statics' = static_values statics ->
  env_rel2 M dead vars vars' -> named_plain2 M dead d = true ->
  exec_named statics hp rcd vars d = exec_named statics' hp rcd vars' d.
Proof.
  intros Hst Hrel Hd.
  destruct d as [s | p args | | slot decls value_slots | | | ];
    simpl in Hd; try discriminate Hd.
  - simpl. rewrite (plain_simple2_eval vars vars' s Hrel Hd). reflexivity.
  - rewrite (exec_named_static_values statics statics' hp rcd vars p args)
      by (intros vs0 sym _ _; rewrite Hst; reflexivity).
    simpl. rewrite (plain_simples2_eval vars vars' args Hrel Hd).
    reflexivity.
  - cbn [exec_named eval_named].
    assert (Hvs : eval_value_slots vars value_slots
                  = eval_value_slots vars' value_slots).
    { induction value_slots as [ | [ws s] value_slots IH];
        [ reflexivity | ].
      simpl in Hd. apply andb_true_iff in Hd as [Hs Hd].
      simpl. rewrite (plain_simple2_eval vars vars' s Hrel Hs), (IH Hd).
      reflexivity. }
    rewrite Hvs. reflexivity.
Qed.

End Envs.

(* ======================================================================== *)
(* Continuations and statics                                                *)
(* ======================================================================== *)

Section Conts.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.
Variable dead : list variable.
Variable CS : code_id -> option code_shape.

Lemma resolve_rel2 (K : finite_map kshape) (conts conts' : cont_env)
    (k : continuation) :
  conts_rel M dead CS K conts conts' ->
  (lookup k K = None ∧ resolve k conts = Outer k
   ∧ resolve k conts' = Outer k)
  ∨ ∃ ksh : kshape,
      lookup k K = Some ksh
      ∧ cont_rel M dead CS ksh (resolve k conts) (resolve k conts').
Proof.
  intros Hrel. unfold resolve.
  induction Hrel as [ | k' sh c c' K conts conts' Hc _ IH]; simpl;
    [ left; split; [ reflexivity | split; reflexivity ] | ].
  rewrite ?lookup_cons. destruct (Pos.eqb k k').
  - right. exists sh. split; [ reflexivity | exact Hc ].
  - exact IH.
Qed.

Lemma rec_closures_rel2_from (KK K S S0 : finite_map kshape)
    (all all' hs hs' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  KK = S ++ K ->
  uc_handlers M dead CS KK all all' S ->
  uc_handlers M dead CS KK hs hs' S0 ->
  env_rel2 M dead vars vars' -> conts_rel M dead CS K conts conts' ->
  conts_rel M dead CS (S0 ++ K)
    (map (fun '(k, h) => (k, Closure h vars conts all)) hs ++ conts)
    (map (fun '(k, h) => (k, Closure h vars' conts' all')) hs' ++ conts').
Proof.
  intros HKK Hall Hhs Hvars Hconts.
  induction Hhs as [ | KK0 k sh ps ps' md md' h h' hs hs' S1 Hps Hmd Hmd' Hh
                       _ IH];
    simpl; [ exact Hconts | ].
  constructor; [ | exact (IH HKK Hall) ].
  subst KK0. econstructor; try eassumption; [ apply transport_refl | ].
  reflexivity.
Qed.

Lemma rec_closures_rel2 (K S : finite_map kshape)
    (handlers handlers' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  uc_handlers M dead CS (S ++ K) handlers handlers' S ->
  env_rel2 M dead vars vars' -> conts_rel M dead CS K conts conts' ->
  conts_rel M dead CS (S ++ K) (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hh.
  exact (rec_closures_rel2_from (S ++ K) K S S _ _ _ _ vars vars' conts
           conts' eq_refl Hh Hh).
Qed.

Lemma lookup_none_keys {A B : Type} (l : list (positive * A))
    (l' : list (positive * B)) (k : positive) :
  map fst l' = map fst l -> lookup k l = None -> lookup k l' = None.
Proof.
  revert l'. induction l as [ | [k0 a] l IH]; intros [ | [k1 b] l'] Hk Hl;
    simpl in Hk; try discriminate Hk; [ reflexivity | ].
  injection Hk as -> Hk. rewrite lookup_cons in Hl. rewrite lookup_cons.
  destruct (Pos.eqb k k0); [ discriminate Hl | exact (IH l' Hk Hl) ].
Qed.

Lemma statics_rel_empty :
  statics_rel M dead CS empty_static_env empty_static_env.
Proof.
  split; [ reflexivity | split; [ reflexivity | ] ].
  intros id c Hc. discriminate Hc.
Qed.

Lemma closure_of_values (S S' : static_env) (v : value) :
  static_values S' = static_values S -> closure_of S' v = closure_of S v.
Proof. intros Hv. destruct v; simpl; try reflexivity. rewrite Hv. reflexivity.
Qed.

(* Bindings related, run in related environments, bind related
   statics. *)
Lemma bindings_eval_rel (bs bs' : list static_binding) :
  uc_bindings M dead CS bs bs' ->
  ∀ (vars vars' : env) (S S' S1 : static_env),
    env_rel2 M dead vars vars' -> statics_rel M dead CS S S' ->
    eval_static_bindings vars bs S = Some S1 ->
    ∃ S1' : static_env,
      eval_static_bindings vars' bs' S' = Some S1'
      ∧ statics_rel M dead CS S1 S1'.
Proof.
  induction 1 as [ | b bs bs' Hb _ IH | id bs bs' _ IH
                 | id c c' cs bs bs' Hframe Hbody _ IH];
    intros vars vars' S S' S1 Hvars (Hval & Hkeys & Hcode) Heval.
  - simpl in Heval. injection Heval as <-. exists S'.
    split; [ reflexivity | split; [ exact Hval | split; assumption ] ].
  - destruct b as [ | | sym [ | | tag fields | | | ]]; simpl in Hb;
      try discriminate Hb.
    simpl in Heval |- *. rewrite Hval.
    destruct (lookup sym (static_values S)); [ discriminate Heval | ].
    simpl in Heval |- *.
    rewrite <- (plain_simples2_eval M dead vars vars' fields Hvars Hb).
    destruct (eval_simples vars fields) as [vs | ]; [ | discriminate Heval ].
    simpl in Heval |- *.
    refine (IH vars vars' _
              (Static_env (static_code S')
                 ((sym, Static_block tag vs) :: static_values S))
              S1 Hvars _ Heval).
    split; [ reflexivity | ].
    split; [ exact Hkeys | exact Hcode ].
  - simpl in Heval |- *.
    exact (IH vars vars' S S' S1 Hvars (conj Hval (conj Hkeys Hcode)) Heval).
  - simpl in Heval |- *.
    destruct (lookup id (static_code S)) eqn:Hl; [ discriminate Heval | ].
    rewrite (lookup_none_keys _ _ id Hkeys Hl).
    refine (IH vars vars' _
              (Static_env ((id, c') :: static_code S') (static_values S'))
              S1 Hvars _ Heval).
    split; [ exact Hval | split ].
    + simpl. rewrite Hkeys. reflexivity.
    + intros id0 c0 Hc0. cbn [static_code] in Hc0 |- *.
      rewrite lookup_cons in Hc0. rewrite lookup_cons.
      destruct (Pos.eqb id0 id) eqn:Hid.
      * apply Pos.eqb_eq in Hid. subst id0.
        injection Hc0 as <-. exists c'. split; [ reflexivity | ].
        exists cs. split; assumption.
      * exact (Hcode id0 c0 Hc0).
Qed.

(* What a call finds to call is a value of kind [Value]. *)
Lemma shape_of_call_value (S : static_env) (v : value) (kind : call_kind)
    (vs : list value) (ra rta : arity) (shp : call_shape) :
  shape_of_call S v kind vs ra rta = Some shp -> kind_of_value v = Value.
Proof.
  destruct v; try reflexivity. intros H.
  unfold shape_of_call, shape_of_closure_call in H.
  destruct kind; cbn in H; discriminate H.
Qed.

(* Code that is called runs or is undefined. *)
Lemma call_code_cases (cl : value) (c : code) (vs : list value)
    (ra rta : arity) (r x : cont_value) (S : static_env) (hp : heap)
    (rd : read_record) :
  call_code cl c vs ra rta r x S hp rd = Undefined
  ∨ ∃ s1 : state, call_code cl c vs ra rta r x S hp rd = Step s1.
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ left; reflexivity | ].
  destruct (negb _); [ left; reflexivity | ].
  destruct (bind_params _ _ _); [ right; eexists; reflexivity | ].
  left; reflexivity.
Qed.

(* A continuation that is no way out is, in the first run, a closure that
   takes values of its kinds. *)
Lemma cont_rel_steps (ksh : kshape) (c c' : cont_value) :
  cont_rel M dead CS ksh c c' -> ks_handler ksh = true ->
  ∀ (vs : list value) (S : static_env) (rd : read_record),
    match apply_cont_value c vs S empty_heap rd with
    | Exit _ _ _ _ => False
    | Step _ => map kind_of_value vs = ks_kinds ksh
    | Undefined => True
    end.
Proof.
  induction 1 as
    [ksh sh0 ps ps' md md' h h' vars vars' K S0 conts conts' rec rec' Hps
       Hfits Hmd Hmd' Hks _ _ _ _
    | ksh k _ Hhandler
    | sh_r sh_k KS b c wps wmd k wargs wvars wconts _ IH _
    | sh_r sh_k KS b cps cmd cargs cvars cconts wps wmd wargs wvars wconts k
        _ _
    | o o' sh KS c c' extra extra' _ _ _ _ _
    | sh KS b cs rem ra c c' e e' _ HKS _ IHc _ _ _];
    intros Hh vs S rd.
  - cbn [apply_cont_value]. rewrite Hmd.
    destruct (kinds_eqb (map kind_of_value vs) (map snd ps)) eqn:Hk;
      [ | exact I ].
    destruct (bind_params _ _ _); [ | exact I ].
    apply kinds_eqb_eq in Hk. rewrite Hk, Hks. reflexivity.
  - rewrite Hhandler in Hh. discriminate Hh.
  - exact (IH eq_refl vs S rd).
  - unfold wrapper_closure. cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds cmd)) eqn:Hk;
      [ | exact I ].
    destruct (bind_params _ _ _); [ | exact I ].
    apply kinds_eqb_eq in Hk. exact Hk.
  - discriminate Hh.
  - (* What is left of an over-application calls the value returned. *)
    cbn [apply_cont_value ks_kinds].
    destruct vs as [ | v [ | v0 vs]]; try exact I.
    destruct (shape_of_call S v Indirect_unknown_arity rem ra
                (cs_result_arity cs))
      as [shp | ] eqn:Hsh; [ | exact I ].
    assert (Hk : map kind_of_value [v] = KS)
      by (cbn [map]; rewrite (shape_of_call_value _ _ _ _ _ _ _ Hsh), HKS;
          reflexivity).
    destruct shp as [cl cc ws aa | cl cc ws rest aa ra1 | p];
      cbn [run_call].
    + destruct (call_code_cases cl cc ws aa (cs_result_arity cs) c e S
                  empty_heap rd)
        as [-> | (s1 & ->)]; [ exact I | exact Hk ].
    + destruct (call_code_cases cl cc ws aa (code_result_arity cc)
                  (Over_application rest ra1 (cs_result_arity cs) c e) e S
                  empty_heap rd)
        as [-> | (s1 & ->)]; [ exact I | exact Hk ].
    + specialize (IHc eq_refl [p] S rd).
      destruct (apply_cont_value c [p] S empty_heap rd);
        [ exact Hk | exact IHc | exact I ].
Qed.

(* A continuation with extra arguments is applied to the values followed
   by them. *)
Lemma apply_with_extra_args (c : cont_value) (extra vs : list value)
    (S : static_env) (hp : heap) (rd : read_record) :
  apply_cont_value (with_extra_args c extra) vs S hp rd
  = apply_cont_value c (vs ++ extra) S hp rd.
Proof.
  destruct extra as [ | w extra]; [ | reflexivity ].
  cbn [with_extra_args]. rewrite app_nil_r. reflexivity.
Qed.

(* The exception, of one shape, followed by the extra arguments. *)
Lemma vrel_one_extra (o : pshape) (sh : list pshape) (v : value)
    (vs' extra extra' : list value) :
  vrel [o] [v] vs' -> vrel sh extra extra' ->
  vrel (o :: sh) (v :: extra) (vs' ++ extra').
Proof.
  intros H He. inversion H; subst;
    match goal with H0 : vrel [] [] _ |- _ => inversion H0; subst end;
    cbn [app]; rewrite ?app_nil_r; constructor; assumption.
Qed.

Lemma eval_simples_len (vars : env) (args : list simple) (vs : list value) :
  eval_simples vars args = Some vs -> length vs = length args.
Proof.
  revert vs. induction args as [ | s args IH]; intros vs Hev; simpl in Hev.
  - injection Hev as <-. reflexivity.
  - destruct (eval_simple vars s); [ | discriminate Hev ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws;
      [ | discriminate Hev ].
    injection Hev as <-. simpl. rewrite (IH ws eq_refl). reflexivity.
Qed.

Lemma forall2_fits_plain (sh sh' : list pshape) :
  Forall2 fits sh sh' -> Forall (fun o => o = Psh_plain) sh' ->
  Forall (fun o => o = Psh_plain) sh.
Proof.
  induction 1 as [ | a b sh sh' Hab _ IH]; intros Hall; [ constructor | ].
  inversion Hall as [ | b0 l0 Hb Hl]; subst. constructor; [ | exact (IH Hl) ].
  inversion Hab; reflexivity.
Qed.

(* A continuation is related at any shapes that fit its own. *)
Lemma cont_rel_refit_fits (ksh : kshape) (sh_r : list pshape)
    (c c' : cont_value) :
  cont_rel M dead CS ksh c c' -> Forall2 fits sh_r (ks_shapes ksh) ->
  cont_rel M dead CS (Kshape sh_r (ks_kinds ksh) (ks_handler ksh)) c c'.
Proof.
  intros Hc. destruct Hc as
    [ksh sh0 ps ps' md md' h h' vars vars' K S0 conts conts' rec rec' Hps
       Hfits Hmd Hmd' Hks Hhb Hvars Hconts Hrec
    | ksh k Hplain Hhandler
    | sh sh_k KS b0 c wps wmd k wargs wvars wconts Hin Hok
    | sh sh_k KS b0 cps cmd cargs cvars cconts wps wmd wargs wvars wconts k
        Hin Hok
    | o o' sh KS c c' extra extra' Hin Hf0 Hl Hve
    | sh KS b0 cs rem ra c c' e e' Htr HKS Hc He Hind];
    intros Hf; cbn [ks_shapes ks_kinds ks_handler] in Hf |- *.
  - apply (cont_rel_closure M dead CS
             (Kshape sh_r (ks_kinds ksh) (ks_handler ksh)) sh0 ps ps' md md'
             h h' vars vars' K S0 conts conts' rec rec');
      try assumption.
    cbn [ks_shapes ks_kinds].
    exact (transport_trans _ _ _ _ (transport_fits _ _ _ Hf) Hfits).
  - apply cont_rel_outer; [ exact (forall2_fits_plain _ _ Hf Hplain) | ].
    exact Hhandler.
  - eapply cont_rel_wrap_u; [ exact Hin | ].
    intros vs vs' Hv Hk. apply Hok; [ exact (vrel_fits _ _ _ Hv _ Hf) | ].
    exact Hk.
  - eapply cont_rel_wrap_both; [ exact Hin | ].
    intros vs vs' Hv Hk. apply Hok; [ exact (vrel_fits _ _ _ Hv _ Hf) | ].
    exact Hk.
  - inversion Hf as [ | o1 o2 l1 l2 Ho Hnil]; subst.
    inversion Hnil; subst.
    exact (cont_rel_extra M dead CS o1 o' sh KS c c' extra extra' Hin
             (fits_trans _ _ _ Ho Hf0) Hl Hve).
  - apply cont_rel_over; try assumption.
    exact (transport_trans _ _ _ _ (transport_fits _ _ _ Hf) Htr).
Qed.

End Conts.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Section Simulation.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.
Variable dead : list variable.
Variable CS : code_id -> option code_shape.

Abbreviation R := (related2 M dead CS).

(* A handler closure applied to values related at its shapes. *)
Lemma closure_answered (sh sh0 : list pshape)
    (ps ps' : list (variable * flambda_kind)) (md md' : handler_metadata)
    (h h' : expr) (vars vars' : env) (K S0 : finite_map kshape)
    (conts conts' : cont_env)
    (rec rec' : list (continuation * continuation_handler))
    (vs vs' : list value) (S S' : static_env) (rd : read_record)
    (s1 : state) :
  params_rel2 M dead sh0 ps ps' -> transport sh sh0 (map snd ps) ->
  hm_params_kinds md = map snd ps -> hm_params_kinds md' = map snd ps' ->
  uc M dead CS (S0 ++ K) h h' -> env_rel2 M dead vars vars' ->
  conts_rel M dead CS K conts conts' ->
  uc_handlers M dead CS (S0 ++ K) rec rec' S0 ->
  vrel sh vs vs' -> statics_rel M dead CS S S' ->
  apply_cont_value (Closure (Handler (map fst ps) md h) vars conts rec) vs S
    empty_heap rd
  = Step s1 ->
  ∃ t1 : state,
    apply_cont_value (Closure (Handler (map fst ps') md' h') vars' conts' rec')
      vs' S' empty_heap rd
    = Step t1
    ∧ related_main M dead CS s1 t1.
Proof.
  intros Hps Hfits Hmd Hmd' Hh Hvars Hconts Hrec Hv Hst Ha.
  cbn [apply_cont_value] in Ha. rewrite Hmd in Ha.
  destruct (kinds_eqb (map kind_of_value vs) (map snd ps)) eqn:Hk;
    [ | discriminate Ha ].
  apply kinds_eqb_eq in Hk.
  destruct (bind_params (map fst ps) vs vars) as [hv | ] eqn:Hb;
    [ | discriminate Ha ].
  injection Ha as <-.
  pose proof (Hfits _ _ Hv Hk) as Hv0.
  pose proof (vrel_kinds_params M dead _ _ _ Hv0 ps ps' Hps Hk) as Hk'.
  destruct (bind_vrel M M_distinct dead _ _ _ Hv0 ps ps' vars vars' hv Hps
              Hvars Hb)
    as (hv' & Hb' & Hrel).
  eexists. split.
  - cbn [apply_cont_value]. rewrite Hmd', <- Hk', kinds_eqb_refl, Hb'.
    reflexivity.
  - exists (S0 ++ K). cbn [state_expr state_vars state_conts state_statics
                           state_heap state_reads].
    split; [ exact Hh | split; [ exact Hrel | ] ].
    split; [ exact (rec_closures_rel2 M dead CS K S0 rec rec' vars vars'
                      conts conts' Hrec Hvars Hconts) | ].
    split; [ exact Hst | repeat split ].
Qed.

Lemma run_one (t t1 : state) : step t = Step t1 -> run 1 t = Step t1.
Proof. intros H. rewrite run_succ, H. reflexivity. Qed.

Lemma resolve_head (k : continuation) (c : cont_value) (conts : cont_env) :
  resolve k ((k, c) :: conts) = c.
Proof. unfold resolve. rewrite lookup_head. reflexivity. Qed.

Lemma resolve_other (k k' : continuation) (c : cont_value)
    (conts : cont_env) :
  k <> k' -> resolve k ((k', c) :: conts) = resolve k conts.
Proof. intros H. unfold resolve. rewrite lookup_other by exact H. reflexivity.
Qed.

(* A continuation that is no way out is related at any shapes that fit
   its own. *)
Lemma cont_rel_refit_gen (ksh : kshape) (sh_r : list pshape) (b : bool)
    (c c' : cont_value) :
  cont_rel M dead CS ksh c c' -> ks_handler ksh = true ->
  transport sh_r (ks_shapes ksh) (ks_kinds ksh) ->
  cont_rel M dead CS (Kshape sh_r (ks_kinds ksh) b) c c'.
Proof.
  intros Hc. destruct Hc as
    [ksh sh0 ps ps' md md' h h' vars vars' K S0 conts conts' rec rec' Hps
       Hfits Hmd Hmd' Hks Hhb Hvars Hconts Hrec
    | ksh k _ Hhandler
    | sh sh_k KS b0 c wps wmd k wargs wvars wconts Hin Hok
    | sh sh_k KS b0 cps cmd cargs cvars cconts wps wmd wargs wvars wconts k
        Hin Hok
    | o o' sh KS c c' extra extra' Hin Hf0 Hl Hve
    | sh KS b0 cs rem ra c c' e e' Htr HKS Hc He Hind];
    intros Hh Hf; cbn [ks_shapes ks_kinds ks_handler] in Hh, Hf |- *.
  - apply (cont_rel_closure M dead CS (Kshape sh_r (ks_kinds ksh) b) sh0 ps
             ps' md md' h h' vars vars' K S0 conts conts' rec rec');
      try assumption.
    cbn [ks_shapes ks_kinds]. exact (transport_trans _ _ _ _ Hf Hfits).
  - rewrite Hhandler in Hh. discriminate Hh.
  - eapply cont_rel_wrap_u; [ exact Hin | ].
    intros vs vs' Hv Hk. apply Hok; [ exact (Hf _ _ Hv Hk) | exact Hk ].
  - eapply cont_rel_wrap_both; [ exact Hin | ].
    intros vs vs' Hv Hk. apply Hok; [ | exact Hk ].
    apply Hf; [ exact Hv | apply kinds_eqb_eq; exact Hk ].
  - discriminate Hh.
  - apply cont_rel_over; try assumption.
    exact (transport_trans _ _ _ _ Hf Htr).
Qed.

Lemma cont_rel_refit (sh sh_r : list pshape) (KS : list flambda_kind)
    (b : bool) (c c' : cont_value) :
  cont_rel M dead CS (Kshape sh KS true) c c' -> transport sh_r sh KS ->
  cont_rel M dead CS (Kshape sh_r KS b) c c'.
Proof.
  intros Hc Hf. exact (cont_rel_refit_gen _ sh_r b c c' Hc eq_refl Hf).
Qed.

(* What the first run comes to, answered by the second: an exit by the
   same exit, a step by steps to a related state. *)
Definition answers (r r' : step_result) : Prop :=
  match r with
  | Undefined => True
  | Exit k ws hp _ => ∃ rd1' : read_record, r' = Exit k ws hp rd1'
  | Step s1 =>
    ∃ (t0 : state) (n : nat) (t1 : state),
      r' = Step t0 ∧ run n t0 = Step t1 ∧ R s1 t1
  end.

Lemma vrel_plain_refl (sh : list pshape) (vs : list value) :
  Forall (fun o => o = Psh_plain) sh -> vrel sh vs vs.
Proof.
  intros Hall. revert vs. induction Hall as [ | o sh Ho _ IH];
    intros [ | v vs]; subst; constructor; auto.
Qed.

Lemma vrel_no_unboxed_refl (sh : list pshape) (vs : list value) :
  no_unboxed sh -> vrel sh vs vs.
Proof.
  revert vs. induction sh as [ | o sh IH]; intros [ | v vs] Hnu;
    [ constructor | constructor | constructor | ].
  assert (Hnu' : no_unboxed sh)
    by (intros fields Hin; exact (Hnu fields (or_intror Hin))).
  destruct o as [ | kind | fields].
  - constructor. exact (IH vs Hnu').
  - constructor; [ left; reflexivity | exact (IH vs Hnu') ].
  - exfalso. exact (Hnu fields (or_introl eq_refl)).
Qed.

(* A call whose callee is a partial application calls its closure with
   the arguments it holds; in both runs, as the statics agree on
   closures. *)
Lemma shape_of_call_cases (S S' : static_env) (v : value)
    (kind : call_kind) (vs : list value) (ra rta : arity) :
  shape_of_call S v kind vs ra rta = None
  ∨ ∃ (cl : value) (kind0 : call_kind) (vs0 : list value) (ra0 : arity),
      shape_of_call S v kind vs ra rta
      = shape_of_closure_call S cl kind0 vs0 ra0 rta
      ∧ shape_of_call S' v kind vs ra rta
        = shape_of_closure_call S' cl kind0 vs0 ra0 rta.
Proof.
  unfold shape_of_call. destruct kind as [id | | code_ids].
  - right. exists v, (Direct id), vs, ra.
    destruct (partial_application_of v) as [[[cl applied] aa] | ];
      split; reflexivity.
  - destruct (partial_application_of v) as [[[cl applied] aa] | ].
    + destruct ra as [ | a ra]; [ left; reflexivity | ].
      destruct (kinds_eqb _ _); [ | left; reflexivity ].
      right. eexists _, _, _, _. split; reflexivity.
    + right. exists v, Indirect_unknown_arity, vs, ra. split; reflexivity.
  - right. exists v, (Indirect_known_arity code_ids), vs, ra.
    destruct (partial_application_of v) as [[[cl applied] aa] | ];
      split; reflexivity.
Qed.

(* Code of the same arities is called alike. *)
Lemma shape_code_swap (cl : value) (c c' : code) (vs : list value)
    (ra rta : arity) :
  code_params_arity c' = code_params_arity c ->
  code_result_arity c' = code_result_arity c ->
  code_is_tupled c' = code_is_tupled c ->
  shape_of_code_call cl c' vs ra rta
  = match shape_of_code_call cl c vs ra rta with
    | Some (Full_call cl0 _ vs0 aa) => Some (Full_call cl0 c' vs0 aa)
    | Some (Over_call cl0 _ vs0 rest aa ra0) =>
      Some (Over_call cl0 c' vs0 rest aa ra0)
    | other => other
    end.
Proof.
  intros Hp Hr Ht. unfold shape_of_code_call. cbv zeta. rewrite Hp, Hr, Ht.
  destruct (Nat.eqb _ _); [ reflexivity | ].
  destruct (negb _); [ reflexivity | ].
  destruct (Nat.ltb _ _); [ reflexivity | ].
  destruct (code_is_tupled c); reflexivity.
Qed.

Lemma shape_full_code (cl : value) (c : code) (vs : list value)
    (ra rta : arity) (cl0 : value) (c0 : code) (vs0 : list value)
    (aa : arity) :
  shape_of_code_call cl c vs ra rta = Some (Full_call cl0 c0 vs0 aa) ->
  c0 = c.
Proof.
  unfold shape_of_code_call. cbv zeta.
  destruct (Nat.eqb _ _);
    [ intros H; injection H; intros; subst; reflexivity | ].
  destruct (negb _); [ intros H; discriminate H | ].
  destruct (Nat.ltb _ _); [ intros H; discriminate H | ].
  destruct (code_is_tupled c); intros H; discriminate H.
Qed.

Lemma shape_over_code (cl : value) (c : code) (vs : list value)
    (ra rta : arity) (cl0 : value) (c0 : code) (vs0 rest : list value)
    (aa ra0 : arity) :
  shape_of_code_call cl c vs ra rta
  = Some (Over_call cl0 c0 vs0 rest aa ra0) ->
  c0 = c ∧ code_result_arity c = [Singleton Value].
Proof.
  unfold shape_of_code_call. cbv zeta.
  destruct (Nat.eqb _ _); [ intros H; discriminate H | ].
  destruct (negb _) eqn:Htest; [ intros H; discriminate H | ].
  destruct (Nat.ltb (length (code_params_arity c)) (length ra)) eqn:Hlt;
    [ | destruct (code_is_tupled c); intros H; discriminate H ].
  intros H. injection H as _ Hc0 _ _ _ _.
  split; [ symmetry; exact Hc0 | ].
  apply negb_false_iff, andb_true_iff in Htest as [_ Htest].
  unfold partial_or_over_application in Htest. cbv zeta in Htest.
  rewrite Hlt in Htest.
  apply andb_true_iff in Htest as [_ Htest].
  destruct (code_result_arity c) as [ | [k | comps] [ | a rest']];
    cbn in Htest; try discriminate Htest.
  all: destruct k; cbn in Htest; try discriminate Htest.
  reflexivity.
Qed.

(* Code that keeps its calling convention, called alike in both runs, with
   the same arguments: its body is entered in related states. *)
Lemma call_code_same (id : code_id) (c c' : code) (cs' : code_shape)
    (cl : value) (vs : list value) (ra rta : arity)
    (r r' x x' : cont_value) (S S' : static_env) (rd : read_record)
    (s1 : state) :
  code_frame M dead CS id c c' cs' ->
  uc M dead CS (code_K c cs') (code_body c) (code_body c') ->
  cs_keeps cs' = true -> Forall (fun o => o = Psh_plain) (cs_params cs') ->
  (rta = code_result_arity c ->
   cont_rel M dead CS (Kshape (cs_results cs') (cs_result_kinds cs') true) r
     r') ->
  cont_rel M dead CS (Kshape [cs_exn cs'] [Value] false) x x' ->
  statics_rel M dead CS S S' ->
  call_code cl c vs ra rta r x S empty_heap rd = Step s1 ->
  ∃ t1 : state,
    call_code cl c' vs ra rta r' x' S' empty_heap rd = Step t1
    ∧ related_main M dead CS s1 t1.
Proof.
  intros Hframe Hbody Hkeeps Hppl Hr Hx Hst Hcall.
  destruct Hframe as (Hcs' & Hpa & Hra' & Hlenc & Hkeepa & Htup & Htup' & Hmc
                      & Hmcf & Hmcv & Hret_eq & Hexn_eq' & Hbs & sh & pk & pk'
                      & Hpk & Hpk' & Hps & Hfits).
  destruct (Hkeepa Hkeeps) as [Hpa1 Hra1].
  unfold call_code in Hcall |- *. rewrite Hbs.
  destruct (binds_statics (code_body c)); [ discriminate Hcall | ].
  rewrite Hpa, Hra', <- Hpa1, <- Hra1.
  destruct (negb
              (kinds_eqb (map kind_of_value vs) (unarize ra)
               && kinds_per_parameter_eqb (unarize_per_parameter ra)
                    (unarize_per_parameter (code_params_arity c))
               && arity_eqb rta (code_result_arity c)))
    eqn:Htest; [ discriminate Hcall | ].
  apply negb_false_iff, andb_true_iff in Htest as [_ Hrta].
  apply arity_eqb_eq in Hrta.
  rewrite Hpk in Hcall. rewrite Hpk', Hmc.
  destruct (bind_params (map fst pk) vs [(code_my_closure c, cl)])
    as [hv | ] eqn:Hb; [ | discriminate Hcall ].
  injection Hcall as <-.
  assert (Henv0 : env_rel2 M dead [(code_my_closure c, cl)]
                    [(code_my_closure c, cl)])
    by exact (env_rel2_bind_same M dead [] [] _ cl (env_rel2_empty M dead)
                Hmcf Hmcv).
  destruct (bind_vrel M M_distinct dead _ _ _
              (vrel_fits _ _ _ (vrel_plain_refl _ vs Hppl) _ Hfits)
              pk pk' _ _ hv Hps Henv0 Hb)
    as (hv' & Hb' & Hrel).
  rewrite Hb'. eexists. split; [ reflexivity | ].
  exists (code_K c cs').
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads].
  split; [ exact Hbody | ].
  split; [ exact Hrel | ].
  split; [ | split; [ exact Hst | repeat split ] ].
  rewrite Hret_eq, Hexn_eq'. unfold code_K.
  constructor; [ exact (Hr Hrta) | constructor; [ exact Hx | constructor ] ].
Qed.

Lemma call_code_no_exit (cl : value) (c : code) (vs : list value)
    (ra rta : arity) (r x : cont_value) (S : static_env) (hp : heap)
    (rd : read_record) (k : continuation) (ws : list value) (hp1 : heap)
    (rd1 : read_record) :
  call_code cl c vs ra rta r x S hp rd <> Exit k ws hp1 rd1.
Proof.
  destruct (call_code_cases cl c vs ra rta r x S hp rd) as [-> | (s1 & ->)];
    discriminate.
Qed.

(* A call of a closure whose code is not known, made at the shape [cs],
   with the same arguments in both runs. Its code keeps its calling
   convention: a full call enters it in related states, the continuation
   of the call taking its results; an over-application enters it, with
   what is left of the over-application taking its one plain result; a
   partial application goes where the call returns. *)
Lemma closure_call_answered (cs : code_shape) (cl : value)
    (kind0 : call_kind) (vs0 : list value) (ra0 : arity)
    (c c' e e' : cont_value) (S S' : static_env) (rd : read_record) :
  ind_targets CS cs ->
  cont_rel M dead CS (Kshape (cs_results cs) (cs_result_kinds cs) true) c c' ->
  cont_rel M dead CS (Kshape [cs_exn cs] [Value] false) e e' ->
  (∀ (ws ws' : list value) (rd0 : read_record),
     vrel (cs_results cs) ws ws' ->
     answers (apply_cont_value c ws S empty_heap rd0)
       (apply_cont_value c' ws' S' empty_heap rd0)) ->
  statics_rel M dead CS S S' ->
  answers
    (run_call (fun p => apply_cont_value c [p] S empty_heap rd)
       (shape_of_closure_call S cl kind0 vs0 ra0 (cs_result_arity cs))
       (cs_result_arity cs) c e S empty_heap rd)
    (run_call (fun p => apply_cont_value c' [p] S' empty_heap rd)
       (shape_of_closure_call S' cl kind0 vs0 ra0 (cs_result_arity cs))
       (cs_result_arity cs) c' e' S' empty_heap rd).
Proof.
  intros Hind Hc He Hans Hst.
  pose proof Hind as (Hcomp & Hpl & Hnu & Hrk).
  pose proof Hst as (Hval & Hkeys & Hcode).
  unfold shape_of_closure_call. cbv zeta.
  rewrite (closure_of_values S S' cl Hval).
  destruct (closure_of S cl) as [[called slots] | ]; [ | exact I ].
  (* The tests of a call of known arity read the same code in both. *)
  assert (Hkf : known_arity_full S' called ra0
                = known_arity_full S called ra0).
  { unfold known_arity_full.
    destruct (lookup called (static_code S)) as [cc | ] eqn:Hcc.
    - destruct (Hcode called cc Hcc) as (cc' & Hcc' & cs' & Hframe & _).
      rewrite Hcc'.
      destruct Hframe as (_ & Hpa' & _ & Hlen & _ & Htup1 & Htup1' & _).
      destruct ra0 as [ | a0 ra0]; [ reflexivity | ]. cbv beta iota.
      rewrite Htup1, Htup1', Hpa', Hlen. reflexivity.
    - rewrite (lookup_none_keys _ _ called Hkeys Hcc). reflexivity. }
  rewrite Hkf.
  destruct (match kind0 with
            | Direct named => Pos.eqb called named
            | Indirect_unknown_arity => true
            | Indirect_known_arity code_ids =>
              code_ids_allow code_ids called
              && known_arity_full S called ra0
            end); [ | exact I ].
  destruct (lookup called (static_code S)) as [cc | ] eqn:Hcc;
    [ | exact I ].
  destruct (Hcode called cc Hcc) as (cc' & Hcc' & cs' & Hframe & Hbody).
  rewrite Hcc'.
  pose proof Hframe as (Hcs' & Hpa & Hra' & _ & Hkeepa & Htup & Htup' & _).
  destruct (Hcomp called cs' Hcs')
    as (Hkeeps & Hppl & Hrk' & Hexf & Hresf & Hover).
  destruct (Hkeepa Hkeeps) as [Hpa1 Hra1].
  unfold arguments_for_code. rewrite Htup, Htup'. cbn beta iota.
  rewrite (shape_code_swap cl cc cc' vs0 ra0 (cs_result_arity cs))
    by congruence.
  assert (Hx : cont_rel M dead CS (Kshape [cs_exn cs'] [Value] false) e e')
    by exact (cont_rel_refit_fits M dead CS _ [cs_exn cs'] e e' He
                (Forall2_cons _ _ Hexf (Forall2_nil _))).
  destruct (shape_of_code_call cl cc vs0 ra0 (cs_result_arity cs))
    as [[cl0 c0 ws aa | cl0 c0 ws rest aa ra1 | p] | ] eqn:Hshape;
    [ | | | exact I ].
  - (* A full call. *)
    rewrite (shape_full_code _ _ _ _ _ _ _ _ _ Hshape). cbn [run_call].
    destruct (call_code cl0 cc ws aa (cs_result_arity cs) c e S empty_heap rd)
      as [s1 | k ws1 hp1 rd1 | ] eqn:Hcall;
      [ | destruct (call_code_no_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcall)
        | exact I ].
    destruct (call_code_same called cc cc' cs' cl0 ws aa (cs_result_arity cs)
                c c' e e' S S' rd s1 Hframe Hbody Hkeeps Hppl)
      as (t1 & Ht1 & Hrel); [ | exact Hx | exact Hst | exact Hcall | ].
    + intros Heq.
      assert (Hres : cs_result_arity cs' = cs_result_arity cs)
        by congruence.
      rewrite Hrk', Hres, <- Hrk.
      exact (cont_rel_refit _ _ _ true c c' Hc
               (transport_fits _ _ _ (Hresf Hres))).
    + exists t1, 0, t1. split; [ exact Ht1 | split; [ reflexivity | ] ].
      left. exact Hrel.
  - (* An over-application. *)
    destruct (shape_over_code _ _ _ _ _ _ _ _ _ _ _ Hshape) as [-> Hone].
    cbn [run_call]. rewrite Hra', <- Hra1.
    destruct (call_code cl0 cc ws aa (code_result_arity cc)
                (Over_application rest ra1 (cs_result_arity cs) c e) e S
                empty_heap rd)
      as [s1 | k ws1 hp1 rd1 | ] eqn:Hcall;
      [ | destruct (call_code_no_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hcall)
        | exact I ].
    destruct (call_code_same called cc cc' cs' cl0 ws aa (code_result_arity cc)
                (Over_application rest ra1 (cs_result_arity cs) c e)
                (Over_application rest ra1 (cs_result_arity cs) c' e') e e' S
                S' rd s1 Hframe Hbody Hkeeps Hppl)
      as (t1 & Ht1 & Hrel); [ | exact Hx | exact Hst | exact Hcall | ].
    + intros _.
      assert (Hres : cs_result_arity cs' = [Singleton Value]) by congruence.
      rewrite (Hover Hres), Hrk', Hres.
      apply cont_rel_over; try assumption; [ apply transport_refl | ].
      reflexivity.
    + exists t1, 0, t1. split; [ exact Ht1 | split; [ reflexivity | ] ].
      left. exact Hrel.
  - (* A partial application. *)
    cbn [run_call].
    exact (Hans [p] [p] rd (vrel_no_unboxed_refl _ _ Hnu)).
Qed.

(* A call whose code is not known, made at the shape [cs]. *)
Lemma ind_call_answered (cs : code_shape) (kind : call_kind) (v : value)
    (vs : list value) (ra : arity) (c c' e e' : cont_value)
    (S S' : static_env) (rd : read_record) :
  ind_targets CS cs ->
  cont_rel M dead CS (Kshape (cs_results cs) (cs_result_kinds cs) true) c c' ->
  cont_rel M dead CS (Kshape [cs_exn cs] [Value] false) e e' ->
  (∀ (ws ws' : list value) (rd0 : read_record),
     vrel (cs_results cs) ws ws' ->
     answers (apply_cont_value c ws S empty_heap rd0)
       (apply_cont_value c' ws' S' empty_heap rd0)) ->
  statics_rel M dead CS S S' ->
  answers
    (run_call (fun p => apply_cont_value c [p] S empty_heap rd)
       (shape_of_call S v kind vs ra (cs_result_arity cs))
       (cs_result_arity cs) c e S empty_heap rd)
    (run_call (fun p => apply_cont_value c' [p] S' empty_heap rd)
       (shape_of_call S' v kind vs ra (cs_result_arity cs))
       (cs_result_arity cs) c' e' S' empty_heap rd).
Proof.
  intros Hind Hc He Hans Hst.
  destruct (shape_of_call_cases S S' v kind vs ra (cs_result_arity cs))
    as [Hn | (cl & kind0 & vs0 & ra0 & H1 & H2)];
    [ rewrite Hn; exact I | ].
  rewrite H1, H2.
  exact (closure_call_answered cs cl kind0 vs0 ra0 c c' e e' S S' rd Hind Hc
           He Hans Hst).
Qed.

(* A call that returns to a continuation that is no way out does not
   end the run. *)
Lemma run_call_no_exit (sh : list pshape) (KS : list flambda_kind)
    (shp : option call_shape) (rta : arity) (c c' e : cont_value)
    (S : static_env) (rd : read_record) (k : continuation)
    (ws : list value) (hp : heap) (rd1 : read_record) :
  cont_rel M dead CS (Kshape sh KS true) c c' ->
  run_call (fun p => apply_cont_value c [p] S empty_heap rd) shp rta c e S
    empty_heap rd
  <> Exit k ws hp rd1.
Proof.
  intros Hc Hex.
  destruct shp as [[cl cc ws0 aa | cl cc ws0 rest aa ra1 | p] | ];
    cbn [run_call] in Hex; try discriminate Hex.
  - exact (call_code_no_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hex).
  - exact (call_code_no_exit _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hex).
  - pose proof (cont_rel_steps M dead CS _ _ _ Hc eq_refl [p] S rd) as H.
    rewrite Hex in H. exact H.
Qed.


(* A continuation applied to values related at its shapes: an exit is
   one of the other, and a step is answered by one step or more. *)
Lemma cont_answered (ksh : kshape) (c c' : cont_value) :
  cont_rel M dead CS ksh c c' ->
  ∀ (vs vs' : list value) (S S' : static_env) (rd : read_record),
    vrel (ks_shapes ksh) vs vs' -> statics_rel M dead CS S S' ->
    match apply_cont_value c vs S empty_heap rd with
    | Undefined => True
    | Exit k ws hp rd1 =>
      ∃ rd1' : read_record,
        apply_cont_value c' vs' S' empty_heap rd = Exit k ws hp rd1'
    | Step s1 =>
      ∃ (t0 : state) (n : nat) (t1 : state),
        apply_cont_value c' vs' S' empty_heap rd = Step t0
        ∧ run n t0 = Step t1 ∧ R s1 t1
    end.
Proof.
  intros Hc. induction Hc as
    [ksh sh0 ps ps' md md' h h' vars vars' K S0 conts conts' rec rec' Hps
       Hfits Hmd Hmd' Hks Hh Hvars Hconts Hrec
    | ksh k Hplain Hhandler
    | sh_r sh_k KS b c wps wmd k wargs wvars wconts Hin IH Hok
    | sh_r sh_k KS b cps cmd cargs cvars cconts wps wmd wargs wvars wconts k
        Hin _ Hok
    | o o' sh KS c c' extra extra' Hin IH Hf Hl Hve
    | sh KS b cs rem ra c c' e e' Htr HKS Hc IHc He _ Hind];
    intros vs vs' S S' rd Hv Hst.
  - destruct (apply_cont_value
                (Closure (Handler (map fst ps) md h) vars conts rec) vs S
                empty_heap rd)
      as [s1 | k ws hp rd1 | ] eqn:Ha; [ | | exact I ].
    + rewrite Hks in Hfits.
      destruct (closure_answered _ sh0 ps ps' md md' h h' vars vars' K S0
                  conts conts' rec rec' vs vs' S S' rd s1 Hps Hfits Hmd Hmd'
                  Hh Hvars Hconts Hrec Hv Hst Ha)
        as (t1 & Ht1 & Hrel).
      exists t1, 0, t1. split; [ exact Ht1 | split; [ reflexivity | ] ].
      left. exact Hrel.
    + exfalso. cbn [apply_cont_value] in Ha.
      destruct (kinds_eqb _ _); [ | discriminate Ha ].
      destruct (bind_params _ _ _); discriminate Ha.
  - rewrite (vrel_plain_eq _ _ _ Hv Hplain). cbn [apply_cont_value].
    exists rd. reflexivity.
  - pose proof (cont_rel_steps M dead CS _ _ _ Hin eq_refl vs S rd) as Hsteps.
    destruct (apply_cont_value c vs S empty_heap rd)
      as [s1 | k0 ws hp rd1 | ] eqn:Ha; [ | contradiction | exact I ].
    cbn [ks_kinds] in Hsteps. cbn [ks_shapes] in Hv.
    destruct (Hok vs vs' Hv Hsteps) as [Hwk Hbind].
    destruct (Hbind wvars) as (hv & vs'' & Hb & Hargs & Hv'').
    specialize (IH vs vs'' S S' rd Hv'' Hst). rewrite Ha in IH.
    destruct IH as (t0 & n & t1 & Ht0 & Hn & Hrel).
    eexists _, (Datatypes.S n), t1. split.
    + unfold wrapper_closure. cbn [apply_cont_value]. rewrite Hwk, Hb.
      reflexivity.
    + split; [ | exact Hrel ].
      rewrite run_succ, step_apply_cont. cbn [ac_args ac_continuation].
      rewrite Hargs. unfold apply_continuation.
      change (rec_closures [] wvars wconts ++ wconts) with wconts.
      rewrite Ht0. exact Hn.
  - unfold wrapper_closure at 1. cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds cmd)) eqn:Hk;
      [ | exact I ].
    destruct (bind_params cps vs cvars) as [hv | ] eqn:Hb; [ | exact I ].
    cbn [ks_shapes] in Hv.
    destruct (Hok vs vs' Hv Hk) as [Hwk Hbind].
    destruct (Hbind cvars wvars hv Hb) as (hv' & Hb' & Hargs).
    eexists _, 0, _. split.
    + unfold wrapper_closure. cbn [apply_cont_value]. rewrite Hwk, Hb'.
      reflexivity.
    + split; [ reflexivity | ]. right. right.
      eexists _, _, (Kshape sh_k KS true).
      cbn [state_expr state_vars state_conts state_statics state_heap
           state_reads ac_continuation ac_args].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ reflexivity | split; [ exact Hin | ] ].
      split; [ exact Hargs | split; [ exact Hst | repeat split ] ].
  - (* The exception, followed by the extra arguments. *)
    rewrite !apply_with_extra_args.
    pose proof (cont_rel_steps M dead CS _ _ _ Hin eq_refl (vs ++ extra) S rd)
      as Hsteps.
    destruct (apply_cont_value c (vs ++ extra) S empty_heap rd)
      as [s1 | k0 ws hp rd1 | ] eqn:Ha; [ | contradiction | exact I ].
    cbn [ks_kinds] in Hsteps.
    apply (f_equal (@length flambda_kind)) in Hsteps.
    rewrite length_map, length_app in Hsteps. cbn [length] in Hsteps.
    destruct vs as [ | v [ | v0 vs]]; cbn [length] in Hsteps; [ lia | | lia ].
    cbn [ks_shapes] in Hv.
    pose proof (vrel_fits _ _ _ (vrel_one_extra o sh v vs' extra extra' Hv Hve)
                  (o' :: sh) (Forall2_cons _ _ Hf (fits_list_refl sh)))
      as Hv'.
    specialize (IH (v :: extra) (vs' ++ extra') S S' rd Hv' Hst).
    cbn [app] in Ha. rewrite Ha in IH. exact IH.
  - (* What is left of an over-application: the call of the one value
       returned, with the arguments left, as an indirect call. *)
    cbn [ks_shapes] in Hv.
    destruct vs as [ | v [ | v0 vs]];
      try (cbn [apply_cont_value]; exact I).
    destruct (shape_of_call S v Indirect_unknown_arity rem ra
                (cs_result_arity cs))
      as [shp | ] eqn:Hsh.
    2: { cbn [apply_cont_value]. rewrite Hsh. cbn [run_call]. exact I. }
    assert (Hvs : vs' = [v]).
    { assert (Hk : map kind_of_value [v] = KS)
        by (cbn [map]; rewrite (shape_of_call_value _ _ _ _ _ _ _ Hsh), HKS;
            reflexivity).
      exact (vrel_plain_eq _ _ _ (Htr [v] vs' Hv Hk)
               (Forall_cons _ eq_refl (Forall_nil _))). }
    subst vs'. cbn [apply_cont_value].
    exact (ind_call_answered cs Indirect_unknown_arity v rem ra c c' e e' S S'
             rd Hind Hc
             He (fun ws ws' rd0 Hw => IHc ws ws' S S' rd0 Hw Hst) Hst).
Qed.

(* What a call of the first run comes to, answered by the second: it
   does not end the run, and a step is answered by steps to a related
   state. *)
Definition call_answers (r r' : step_result) : Prop :=
  match r with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ (t0 : state) (n : nat) (t1 : state),
      r' = Step t0 ∧ run n t0 = Step t1 ∧ R s1 t1
  end.

Lemma answers_call (r r' : step_result) :
  answers r r' -> (∀ k ws hp rd1, r <> Exit k ws hp rd1) ->
  call_answers r r'.
Proof.
  destruct r as [s1 | k ws hp rd1 | ]; intros H Hne;
    [ exact H | exact (Hne _ _ _ _ eq_refl) | exact I ].
Qed.

(* A call related is answered by the call of related code: of the code
   it is known to call, or of the code of the closure called. *)
Lemma call_answered (s t : state) :
  related_call M dead CS s t -> call_answers (step s) (step t).
Proof.
  destruct s as [p vars conts S hp rd], t as [q vars' conts' S' hp' rd'].
  intros (ap & ap' & cs & Hp & Hq & Hcall & Hvars & Hret & Hexn & Hst & Hhp
          & Hhp' & Hrd).
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst p q hp hp' rd'.
  destruct Hcall as (Htarget & Hcallee & Hcplain & Hargs & Hnd & Haa & Hra
                     & Hlen & Hkinds & Hexn_eq).
  rewrite !step_apply.
  rewrite Hcallee, <- (plain_simple2_eval M dead vars vars' _ Hvars Hcplain).
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcv;
    [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
    [ | exact I ].
  destruct (args_vrel M dead _ _ _ Hargs vars vars' vs Hvars Hvs)
    as (vs' & Hvs' & Hv).
  rewrite Hvs'.
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ] eqn:Hex;
    [ | exact I ].
  destruct (Hexn extra eq_refl) as (extra' & Hex' & Hexc).
  rewrite Hex'.
  destruct Htarget as [(id & Hkind & Hkind' & Hcs)
                      | (Hkind & Hind & Haa0 & Hra0)].
  2: {
    (* A call whose code is not known. *)
    pose proof Hind as (_ & Hpl & _ & _).
    assert (Hvv : vs' = vs) by exact (vrel_plain_eq _ _ _ Hv Hpl).
    subst vs'.
    rewrite Hkind, Haa, Hra, <- Haa0, Hra0.
    unfold call_function.
    exact (answers_call _ _
             (ind_call_answered cs (ap_call_kind ap) callee vs
                (ap_args_arity ap) _ _ _ _ S S' rd Hind Hret Hexc
                (fun ws ws' rd0 Hw =>
                   cont_answered _ _ _ Hret ws ws' S S' rd0 Hw Hst)
                Hst)
             (fun k ws hp1 rd1 =>
                run_call_no_exit _ _ _ _ _ _ _ _ _ k ws hp1 rd1 Hret)). }
  unfold call_function, run_call, shape_of_call.
  rewrite Hkind, Hkind'. cbn beta iota.
  unfold shape_of_closure_call.
  destruct Hst as (Hval & Hkeys & Hcode).
  rewrite (closure_of_values S S' callee Hval).
  destruct (closure_of S callee) as [[called slots] | ];
    [ | exact I ].
  destruct (Pos.eqb called id) eqn:Hid; [ | exact I ].
  apply Pos.eqb_eq in Hid. subst called.
  destruct (lookup id (static_code S)) as [c | ] eqn:Hc;
    [ | exact I ].
  destruct (Hcode id c Hc) as (c' & Hc' & cs' & Hframe & Hbody).
  rewrite Hc'.
  destruct Hframe as (Hcs' & Hpa & Hra' & Hlenc & _ & Htup & Htup' & Hmc & Hmcf
                      & Hmcv & Hret_eq & Hexn_eq' & Hbs & sh & pk & pk' & Hpk
                      & Hpk' & Hps & Hfits).
  rewrite Hcs in Hcs'. injection Hcs' as <-.
  unfold arguments_for_code. rewrite Htup, Htup'. cbn beta iota.
  unfold shape_of_code_call.
  rewrite Hlenc, Hlen, Nat.eqb_refl, Haa, Hpa, Nat.eqb_refl.
  cbn beta iota. unfold call_code. rewrite Hbs.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct (kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap))
            && kinds_per_parameter_eqb
                 (unarize_per_parameter (ap_args_arity ap))
                 (unarize_per_parameter (code_params_arity c))
            && arity_eqb (ap_return_arity ap) (code_result_arity c)) eqn:Htest;
    [ | exact I ].
  apply andb_true_iff in Htest as [Htest _].
  apply andb_true_iff in Htest as [Htk _].
  apply kinds_eqb_eq in Htk.
  assert (Htest' :
    kinds_eqb (map kind_of_value vs') (unarize (cs_params_arity cs))
    && kinds_per_parameter_eqb
         (unarize_per_parameter (cs_params_arity cs))
         (unarize_per_parameter (code_params_arity c'))
    && arity_eqb (ap_return_arity ap') (code_result_arity c') = true).
  { rewrite Hpa, Hra, Hra',
      (vrel_shape_kinds _ _ _ Hv Hnd), Htk, <- Hkinds, kinds_eqb_refl,
      arity_eqb_refl.
    rewrite (proj2 (kinds_per_parameter_eqb_eq _ _) eq_refl). reflexivity. }
  rewrite Htest'. cbn [negb].
  rewrite Hpk, Hmc.
  destruct (bind_params (map fst pk) vs [(code_my_closure c, callee)])
    as [hv | ] eqn:Hb; [ | exact I ].
  assert (Henv0 : env_rel2 M dead [(code_my_closure c, callee)]
                    [(code_my_closure c, callee)])
    by exact (env_rel2_bind_same M dead [] [] _ callee (env_rel2_empty M dead)
                Hmcf Hmcv).
  destruct (bind_vrel M M_distinct dead _ _ _ (vrel_fits _ _ _ Hv _ Hfits)
              pk pk' _ _ hv Hps Henv0 Hb)
    as (hv' & Hb' & Hrel).
  rewrite Hpk', Hb'.
  eexists _, 0, _. split; [ reflexivity | split; [ reflexivity | ] ].
  left. exists (code_K c cs).
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads].
  split; [ exact Hbody | ].
  split; [ exact Hrel | ].
  split; [ | split; [ exact (conj Hval (conj Hkeys Hcode)) | repeat split ] ].
  rewrite Hret_eq, Hexn_eq'. unfold code_K.
  constructor; [ exact Hret | constructor; [ exact Hexc | constructor ] ].
Qed.

(* A call answered is a step answered. *)
Lemma call_answers_answered (s t : state) :
  call_answers (step s) (step t) -> answered R eq measure s t.
Proof.
  unfold answered, call_answers.
  destruct (step s) as [s1 | k ws hp rd1 | ];
    [ | intros [] | intros _; exact I ].
  intros (t0 & n & t1 & Ht0 & Hn & Hrel).
  destruct n as [ | n].
  - cbn [run] in Hn. injection Hn as <-.
    right. left. exists t0. split; [ exact Ht0 | exact Hrel ].
  - right. right. right. exists t1, n. split; [ | exact Hrel ].
    rewrite run_succ, Ht0. exact Hn.
Qed.



(* The exception continuations of related calls, with their extra
   arguments, are related at the shape of the exceptions of the code. *)
Lemma exn_rel (K : finite_map kshape) (conts conts' : cont_env)
    (vars vars' : env) (o : pshape) (ap ap' : apply_expr) :
  conts_rel M dead CS K conts conts' -> env_rel2 M dead vars vars' ->
  exn_call_rel M dead K o ap ap' ->
  ∀ extra : list value,
    eval_simples vars (exn_extra_simples ap) = Some extra ->
    ∃ extra' : list value,
      eval_simples vars' (exn_extra_simples ap') = Some extra'
      ∧ cont_rel M dead CS (Kshape [o] [Value] false)
          (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
          (with_extra_args (resolve (ap_exn_continuation ap) conts')
             extra').
Proof.
  intros Hconts Hvars He extra Hextra.
  destruct He as [(Hx & Hx' & He) | (o_k & sh & KS & Hl & Hf & Hargs & Hlen)].
  - unfold exn_extra_simples in Hextra |- *. rewrite Hx in Hextra.
    rewrite Hx'. cbn [map eval_simples] in Hextra. injection Hextra as <-.
    exists [].
    split; [ reflexivity | ]. cbn [with_extra_args].
    destruct (resolve_rel2 M dead CS K conts conts' (ap_exn_continuation ap)
                Hconts)
      as [(Hl & -> & ->) | (ksh & Hl & Hc)].
    + destruct He as [(_ & ->) | (o_k & He & _)];
        [ | rewrite He in Hl; discriminate Hl ].
      apply cont_rel_outer; [ repeat constructor | reflexivity ].
    + destruct He as [(He & _) | (o_k & He & Hf)]; rewrite He in Hl;
        [ discriminate Hl | ].
      injection Hl as <-.
      exact (cont_rel_refit_fits M dead CS _ [o] _ _ Hc
               (Forall2_cons _ _ Hf (Forall2_nil _))).
  - destruct (args_vrel M dead _ _ _ Hargs vars vars' extra Hvars Hextra)
      as (extra' & Hextra' & Hve).
    exists extra'. split; [ exact Hextra' | ].
    destruct (resolve_rel2 M dead CS K conts conts' (ap_exn_continuation ap)
                Hconts)
      as [(Hl' & _) | (ksh & Hl' & Hc)]; rewrite Hl in Hl';
      [ discriminate Hl' | ].
    injection Hl' as <-.
    apply (cont_rel_extra M dead CS o o_k sh KS); try assumption.
    rewrite (eval_simples_len _ _ _ Hextra). unfold exn_extra_simples.
    rewrite length_map. exact Hlen.
Qed.

Lemma find_arm_rel2 (K : finite_map kshape) (i : Z)
    (arms arms' : list (Z * apply_cont)) :
  Forall2
    (fun (arm arm' : Z * apply_cont) =>
       fst arm' = fst arm ∧ ac_rel2 M dead K (snd arm) (snd arm'))
    arms arms' ->
  ∀ ac : apply_cont,
    find_arm i arms = Some ac ->
    ∃ ac' : apply_cont, find_arm i arms' = Some ac' ∧ ac_rel2 M dead K ac ac'.
Proof.
  induction 1 as [ | [d ac0] [d' ac0'] arms arms' [Hd Hac] _ IH];
    intros ac Hfind; [ discriminate Hfind | ].
  simpl in Hd, Hac, Hfind |- *. subst d'. destruct (Z.eqb i d).
  - injection Hfind as <-. exists ac0'. split; [ reflexivity | exact Hac ].
  - exact (IH ac Hfind).
Qed.

(* The [Let]s of the variables of the fields of an unboxed variable run,
   and bind them as the relation asks. *)
Lemma lets_case2 (vars vars' : env) (x : variable) (fields : unboxed_fields)
    (w : value) (l : list (variable * flambda_kind * simple)) (body : expr)
    (conts : cont_env) (statics : static_env) (hp : heap) (rcd : read_record) :
  env_rel2 M dead vars vars' -> fields_of M x = Some fields ->
  map lets_variable l = leaf_variables fields ->
  (∀ (v : variable) (kind : flambda_kind) (s : simple) (v' : variable),
     (v, kind, s) ∈ l -> v' ∈ leaf_variables fields -> s <> Var v') ->
  (∀ (v : variable) (kind : flambda_kind) (s : simple),
     (v, kind, s) ∈ l ->
     ∃ w' : value, eval_simple vars' s = Some w' ∧ kind_of_value w' = kind) ->
  (∀ (p : list field) (v : variable) (k : flambda_kind),
     upath Value (Unboxed fields) p v k ->
     ∃ (s : simple) (w' : value),
       (v, k, s) ∈ l ∧ eval_simple vars' s = Some w'
       ∧ path_value p w = Some w') ->
  ∃ vars1 : env,
    run (length l) (State (lets l body) vars' conts statics hp rcd)
    = Step (State body vars1 conts statics hp rcd)
    ∧ env_rel2 M dead ((x, w) :: vars) vars1.
Proof.
  intros Hvars Hx Hvl Hread Heval Hpaths.
  destruct (lets_run l body vars' conts statics hp rcd)
    as (vars1 & Hrun & Hother & Hbound).
  - rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hx).
  - intros v kind s v' Hin Hv'. rewrite Hvl in Hv'.
    exact (Hread v kind s v' Hin Hv').
  - exact Heval.
  - exists vars1. split; [ exact Hrun | ].
    apply (env_rel2_bind_fields M M_distinct dead vars vars' vars1 x fields w
             Hvars Hx).
    + intros z Hz. apply Hother. rewrite Hvl. exact Hz.
    + intros p v k Hp. destruct (Hpaths p v k Hp) as (s & w' & Hin & Hs & Hpw).
      destruct (Heval v k s Hin) as (w'' & Hs' & Hk).
      rewrite Hs in Hs'. injection Hs' as <-.
      exists w'. split; [ exact Hpw | ].
      split; [ exact (Hbound v k s w' Hin Hs) | exact Hk ].
Qed.

Lemma answer_steps (s1 t t0 t1 : state) (n : nat) :
  step t = Step t0 -> run n t0 = Step t1 -> R s1 t1 ->
  (∃ t' : state, step t = Step t' ∧ R s1 t')
  ∨ (∃ (t' : state) (m : nat), run (S (S m)) t = Step t' ∧ R s1 t').
Proof.
  intros Ht Hn Hrel. destruct n as [ | m].
  - simpl in Hn. injection Hn as <-. left. exists t0. split; assumption.
  - right. exists t1, m. rewrite run_succ, Ht. split; [ exact Hn | exact Hrel ].
Qed.

Lemma pending_answered (s t : state) :
  related_pending M dead CS s t -> answered R eq measure s t.
Proof.
  destruct s as [p vars conts S hp rd], t as [q vars' conts' S' hp' rd'].
  intros (ac & ac' & ksh & Hp & Hq & Hk & Hc & Hargs & Hst & Hhp & Hhp' & Hrd).
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst p q hp hp' rd'. unfold answered.
  rewrite step_apply_cont.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs; [ | exact I ].
  destruct (Hargs vs eq_refl) as (vs' & Hvs' & Hv).
  unfold apply_continuation.
  pose proof (cont_answered ksh _ _ Hc vs vs' S S' rd Hv Hst) as Ha.
  destruct (apply_cont_value (resolve (ac_continuation ac) conts) vs S
              empty_heap rd)
    as [s1 | k ws hp1 rd1 | ] eqn:Hs; [ | | exact I ].
  - destruct Ha as (t0 & n & t1 & Ht0 & Hn & Hrel).
    right.
    assert (Ht : step (State (Apply_cont ac') vars' conts' S' empty_heap rd)
                 = Step t0).
    { rewrite step_apply_cont, Hvs'. unfold apply_continuation.
      exact Ht0. }
    destruct (answer_steps s1 _ t0 t1 n Ht Hn Hrel) as [H | H];
      [ left; exact H | right; right; exact H ].
  - destruct Ha as (rd1' & Ha). exists ws, rd1'. split; [ | reflexivity ].
    rewrite step_apply_cont, Hvs'. unfold apply_continuation.
    exact Ha.
Qed.

(* The second run steps into the call through its wrapper, the first
   into the call: both then at related calls. *)
Lemma main_answered (s t : state) :
  related_main M dead CS s t -> answered R eq measure s t.
Proof.
  destruct s as [p vars conts statics hp rcd],
    t as [q vars' conts' statics' hp' rcd'].
  intros (K & Hub & Hvars & Hconts & Hst & Hhp & Hhp' & Hrcd).
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst hp hp' rcd'. unfold answered. revert Hconts.
  destruct Hub as [K x kind d body body' Hx Hd Hnh Hbody
    | K x d fields ls body body' Hfields Hxf Hl Hoff Hbody
    | K x y fields fields_y pairs body body' Hfields Hxf Hy Hne Hl Hbody
    | K x kind d f s y fields fields_y nested pairs body body' Hfields Hxf Hread
        Hs Hy Hne Hin Hl Hbody
    | K x kind d f s y v fields body body' Hx Hread Hs Hfields Hin Hbody
    | K bindings bindings' body body' Hbindings Hbody
    | K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' Hh Hbody
    | K S handlers handlers' body body' Hhandlers Hstat Hbody
    | K ac ac' Hac
    | K scrutinee arms arms' Hscrutinee Harms
    | K
    | K ap ap' cs k sh Hcall Hexn Hk Hk' Hlk Hfits
    | K ap ap' cs k sh w wps wmd wargs Hcall Hexn Hk Hw Hwe Hlk Hok
    | K ap ap' cs k sh KS wc cps cmd cargs w wps wmd wargs Hcall Hexn Hwc Hw
        Hwce Hwe Hlk Hcmd Hok ]; intros Hconts.
  - (* A [Let] of both. *)
    rewrite step_let_exec,
      (named_plain2_exec M dead statics statics' empty_heap rcd vars vars' d
         (proj1 Hst) Hvars Hd).
    destruct (exec_named statics' empty_heap rcd vars' d)
      as [[[w hp1] rcd1] | ] eqn:Hw; [ | exact I ].
    destruct (exec_named_empty_heap statics' rcd vars' d w hp1 rcd1 Hnh Hw)
      as (_ & -> & ->).
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hk; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_exec, Hw, Hk; reflexivity | ].
    left. exists K. split; [ exact Hbody | ].
    split; [ exact (env_rel2_bind M dead _ _ x w Hvars Hx) | ].
    split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* The allocation of an unboxed variable. *)
    rewrite step_let_exec.
    destruct (exec_named statics empty_heap rcd vars d) as [[[w hp1] rcd1] | ]
      eqn:Hx; [ | exact I ].
    destruct (exec_named_empty_heap statics rcd vars d w hp1 rcd1 Hoff Hx)
      as (Hw & -> & ->).
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (lets_case2 vars vars' x fields w (concat ls) body' conts'
                statics' empty_heap rcd Hvars Hfields)
      as (vars1 & Hrun & Hrel).
    + clear -Hl. induction Hl as [ | [f u] l fields ls Hfu _ IH];
        [ reflexivity | ].
      cbn [concat]. rewrite map_app, IH. unfold leaf_variables.
      rewrite leaves_cons, map_app. f_equal.
      destruct u as [v | nested]; cbn [alloc_entry2] in Hfu.
      * destruct Hfu as (kind & s & -> & _). reflexivity.
      * destruct Hfu as (b & fb & pairs & _ & _ & _ & Hpairs & ->).
        rewrite pairs_lets_variables.
        destruct (pairs_spec _ _ _ _ Hpairs) as (_ & _ & Hc & _).
        rewrite Hc. reflexivity.
    + intros v kind s v' Hin Hv' ->.
      destruct (forall2_concat_in _ _ _ _ Hl Hin)
        as ([f u] & l0 & _ & Hent & Hin0).
      destruct u as [v0 | nested]; cbn [alloc_entry2] in Hent.
      * destruct Hent as (kind0 & s0 & -> & _ & Hplain).
        destruct Hin0 as [Heq | []]. injection Heq as <- <- ->.
        simpl in Hplain. apply plain2_iff in Hplain as [Hplain _].
        apply plain_variable_iff in Hplain as [_ Hnot].
        pose proof (field_variables_in M x fields v' Hfields Hv') as Hfv.
        apply among_In in Hfv. rewrite Hfv in Hnot. discriminate.
      * destruct Hent as (b & fb & pairs & _ & Hb & Hxb & Hpairs & ->).
        apply in_pairs_lets in Hin0 as (v'' & Heq & Hin').
        injection Heq as <-.
        destruct (pairs_spec _ _ _ _ Hpairs) as (Ha & _).
        destruct (Ha v v' kind Hin') as (p & _ & Hp').
        exact (leaves_apart M M_distinct b x fb fields v' Hb Hfields
                 (not_eq_sym Hxb) (upath_leaf_in fb p v' kind Hp') Hv').
    + intros v kind s Hin.
      destruct (forall2_concat_in _ _ _ _ Hl Hin)
        as ([f u] & l0 & _ & Hent & Hin0).
      destruct u as [v0 | nested]; cbn [alloc_entry2] in Hent.
      * destruct Hent as (kind0 & s0 & -> & Hf & Hplain).
        destruct Hin0 as [Heq | []]. injection Heq as <- <- <-.
        destruct (alloc_field_eval statics vars d w f kind0 s0 Hw Hf)
          as (w' & Hs & _ & Hkw).
        exists w'. split; [ | exact Hkw ].
        rewrite <- (plain_simple2_eval M dead vars vars' s0 Hvars Hplain).
        exact Hs.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval statics vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & _ & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 Hvars b fb wb Hb Hs))
          as (_ & Hseconds & _).
        apply in_pairs_lets in Hin0 as (v' & -> & Hin').
        destruct (Hseconds v v' kind Hin') as [_ (w' & Hl' & Hk')].
        exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k Hpath.
      apply upath_unboxed_inv in Hpath as (f & u & p0 & -> & Hfu & Hp0).
      destruct (forall2_in_both _ _ _ _ Hl Hfu) as (l0 & Hl0 & Hent).
      destruct u as [v0 | nested]; cbn [alloc_entry2] in Hent.
      * apply upath_not_unboxed_inv in Hp0 as (-> & -> & ->).
        destruct Hent as (kind0 & s0 & -> & Hf & Hplain).
        pose proof (alloc_field_kind d f kind0 s0 Hf) as Hk0. subst kind0.
        destruct (alloc_field_eval statics vars d w f (field_kind f) s0 Hw Hf)
          as (w' & Hs & Hfw & _).
        exists s0, w'. split; [ exact (in_concat_of _ _ _ Hl0 (or_introl
                                                       eq_refl)) | ].
        split; [ rewrite <- (plain_simple2_eval M dead vars vars' s0 Hvars
                               Hplain);
                 exact Hs | ].
        cbn [path_value]. rewrite Hfw. reflexivity.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval statics vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & Hfw & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 Hvars b fb wb Hb Hs))
          as (_ & _ & Hpaths).
        destruct (Hpaths p0 v k (upath_unboxed_kind _ Value _ _ _ _ Hp0))
          as (v' & w' & Hin' & Hl' & Hpw).
        exists (Var v'), w'.
        split; [ exact (in_concat_of _ _ _ Hl0
                          (pairs_in_lets pairs v v' k Hin')) | ].
        split; [ exact Hl' | ]. cbn [path_value]. rewrite Hfw. exact Hpw.
    + apply (run_answer R _ _ _ _ (length (concat ls)) Hrun).
      * left. exists K.
        cbn [state_expr state_vars state_conts state_statics state_heap
             state_reads].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* An alias of an unboxed variable. *)
    rewrite step_let_simple. cbn [eval_simple].
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (pairs_hold vars' fields fields_y w pairs Hl
                (proj2 Hvars y fields_y w Hy Hw))
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case2 vars vars' x fields w (pairs_lets pairs) body' conts'
                statics' empty_heap rcd Hvars Hfields)
      as (vars1 & Hrun & Hrel).
    + rewrite pairs_lets_variables. exact Hfirsts.
    + intros v kind s v' Hin Hv' ->.
      apply in_pairs_lets in Hin as (v'' & Heq & Hin). injection Heq as <-.
      destruct (Hseconds v v' kind Hin) as [Hleaf _].
      exact (leaves_apart M M_distinct y x fields_y fields v' Hy Hfields
               (not_eq_sym Hne) Hleaf Hv').
    + intros v kind s Hin.
      apply in_pairs_lets in Hin as (v' & -> & Hin).
      destruct (Hseconds v v' kind Hin) as [_ (w' & Hl' & Hk')].
      exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k Hpath.
      destruct (Hpaths p v k Hpath) as (v' & w' & Hin & Hl' & Hpw).
      exists (Var v'), w'.
      split; [ exact (pairs_in_lets pairs v v' k Hin) | ].
      split; [ exact Hl' | exact Hpw ].
    + apply (run_answer R _ _ _ _ (length (pairs_lets pairs)) Hrun).
      * left. exists K.
        cbn [state_expr state_vars state_conts state_statics state_heap
             state_reads].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* The load of a field unboxed in turn. *)
    rewrite step_let_exec.
    destruct (exec_named statics empty_heap rcd vars d) as [[[r hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap statics rcd vars d r hp1 rcd1
                (read_not_heap d f s Hread) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r) Value) eqn:Hk; [ | exact I ].
    destruct (read_eval statics vars d f s r Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    pose proof (fields_hold_nested vars' fields_y nested f w0 r
                  (proj2 Hvars y fields_y w0 Hy Hw0) Hin Hfield) as Hhold.
    destruct (pairs_hold vars' fields nested r pairs Hl Hhold)
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case2 vars vars' x fields r (pairs_lets pairs) body' conts'
                statics' empty_heap rcd Hvars Hfields)
      as (vars1 & Hrun & Hrel).
    + rewrite pairs_lets_variables. exact Hfirsts.
    + intros v kind0 s v' Hin0 Hv' ->.
      apply in_pairs_lets in Hin0 as (v'' & Heq & Hin0).
      injection Heq as <-.
      destruct (Hseconds v v' kind0 Hin0) as [Hleaf _].
      exact (leaves_apart M M_distinct y x fields_y fields v' Hy Hfields
               (not_eq_sym Hne)
               (leaves_unboxed_in Value fields_y f (Unboxed nested) v' Hin
                  Hleaf)
               Hv').
    + intros v kind0 s Hin0.
      apply in_pairs_lets in Hin0 as (v' & -> & Hin0).
      destruct (Hseconds v v' kind0 Hin0) as [_ (w' & Hl' & Hk')].
      exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k0 Hpath.
      destruct (Hpaths p v k0 Hpath) as (v' & w' & Hin0 & Hl' & Hpw).
      exists (Var v'), w'.
      split; [ exact (pairs_in_lets pairs v v' k0 Hin0) | ].
      split; [ exact Hl' | exact Hpw ].
    + apply (run_answer R _ _ _ _ (length (pairs_lets pairs)) Hrun).
      * left. exists K.
        cbn [state_expr state_vars state_conts state_statics state_heap
             state_reads].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* A read of a field. *)
    rewrite step_let_exec.
    destruct (exec_named statics empty_heap rcd vars d) as [[[r hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap statics rcd vars d r hp1 rcd1
                (read_not_heap d f s Hread) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r) kind) eqn:Hk; [ | exact I ].
    destruct (read_eval statics vars d f s r Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    assert (Hp : upath Value (Unboxed fields) [f] v (field_kind f))
      by (econstructor; [ exact Hin | constructor ]).
    destruct (proj2 Hvars y fields w0 Hfields Hw0 [f] v _ Hp)
      as (w' & Hpw & Hv & _).
    cbn [path_value] in Hpw.
    destruct (field_value f w0) as [w1 | ] eqn:Hfw; [ | discriminate Hpw ].
    injection Hpw as <-.
    rewrite <- (Hfield w1 eq_refl) in Hv.
    right. left. eexists. split.
    + rewrite step_let_simple. cbn [eval_simple]. rewrite Hv, Hk.
      reflexivity.
    + left. exists K. split; [ exact Hbody | ].
      split; [ exact (env_rel2_bind M dead _ _ x r Hvars Hx) | ].
      split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* A [Let_static]. *)
    rewrite step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [statics1 | ]
      eqn:Hst1; [ | exact I ].
    destruct (bindings_eval_rel M dead CS bindings bindings' Hbindings vars
                vars' statics statics' statics1 Hvars Hst Hst1)
      as (statics1' & Hst1' & Hrel1).
    right. left. eexists.
    split; [ rewrite step_let_static, Hst1'; reflexivity | ].
    left. exists K. split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ exact Hconts | split; [ exact Hrel1 | repeat split ] ].
  - (* A non-recursive [Let_cont]. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    left. exists ((k, Kshape sh (map snd ps) true) :: K).
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    constructor; [ | exact Hconts ].
    apply (cont_rel_closure M dead CS (Kshape sh (map snd ps) true) sh ps ps'
             md md' h h' vars vars' K []
             conts conts' [] [] Hps (transport_refl sh _) Hmd Hmd' eq_refl Hh
             Hvars Hconts (uc_handlers_nil M dead CS K)).
  - (* A recursive [Let_cont]. *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hb; [ exact I | ].
    right. left. eexists.
    split.
    { rewrite step_let_cont_recursive.
      destruct (handlers_bind_statics handlers') eqn:Hb';
        [ congruence | reflexivity ]. }
    left. exists (S ++ K). split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ apply rec_closures_rel2; assumption | ].
    split; [ exact Hst | repeat split ].
  - (* An [Apply_cont]. *)
    destruct Hac as (Hk & Htrap & Hargs).
    apply pending_answered.
    destruct (resolve_rel2 M dead CS K conts conts' (ac_continuation ac)
                Hconts)
      as [(HK & Hc & Hc') | (ksh & HK & Hc)].
    + exists ac, ac', (Kshape [] [] false).
      cbn [state_expr state_vars state_conts state_statics state_heap
           state_reads].
      split; [ reflexivity | split; [ reflexivity | split; [ exact Hk | ] ] ].
      split; [ rewrite Hk, Hc, Hc'; apply cont_rel_outer;
               [ constructor | reflexivity ] | ].
      split; [ | split; [ exact Hst | repeat split ] ].
      unfold kshape_at in Hargs. rewrite HK in Hargs.
      intros vs Hvs. exact (args_vrel M dead _ _ _ Hargs vars vars' vs Hvars
                              Hvs).
    + exists ac, ac', ksh.
      cbn [state_expr state_vars state_conts state_statics state_heap
           state_reads].
      split; [ reflexivity | split; [ reflexivity | split; [ exact Hk | ] ] ].
      split; [ rewrite Hk; exact Hc | ].
      split; [ | split; [ exact Hst | repeat split ] ].
      unfold kshape_at in Hargs. rewrite HK in Hargs.
      intros vs Hvs. exact (args_vrel M dead _ _ _ Hargs vars vars' vs Hvars
                              Hvs).
  - (* A [Switch]. *)
    rewrite step_switch,
      (plain_simple2_eval M dead vars vars' scrutinee Hvars Hscrutinee).
    destruct (eval_simple vars' scrutinee) as [w | ] eqn:Hw; [ | exact I ].
    destruct (naked_immediate_of_value w) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    destruct (find_arm_rel2 K i arms arms' Harms ac Harm)
      as (ac' & Harm' & Hac).
    right. left. eexists.
    split; [ rewrite step_switch, Hw, Hi, Harm'; reflexivity | ].
    left. exists K. split; [ constructor; exact Hac | ].
    split; [ exact Hvars | split; [ exact Hconts | ] ].
    split; [ exact Hst | repeat split ].
  - rewrite step_invalid. exact I.
  - (* A call, without wrappers. *)
    assert (Hc : related_call M dead CS
                   (State (Apply ap) vars conts statics empty_heap rcd)
                   (State (Apply ap') vars' conts' statics' empty_heap rcd)).
    { exists ap, ap', cs.
      cbn [state_expr state_vars state_conts state_statics state_heap
           state_reads].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ exact Hcall | split; [ exact Hvars | ] ].
      split.
      - rewrite Hk, Hk'.
        destruct (resolve_rel2 M dead CS K conts conts' k Hconts)
          as [(HK & _) | (ksh & HK & Hck)]; rewrite Hlk in HK;
          [ discriminate HK | ].
        injection HK as <-. exact (cont_rel_refit _ _ _ true _ _ Hck Hfits).
      - destruct Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & Hexn_eq).
        split; [ | split; [ exact Hst | repeat split ] ].
        intros extra Hextra. rewrite Hexn_eq.
        exact (exn_rel K conts conts' vars vars' _ ap ap' Hconts Hvars Hexn
                 extra Hextra). }
    exact (call_answers_answered _ _ (call_answered _ _ Hc)).
  - (* A call through a wrapper of the second run. *)
    pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & Hexn_eq).
    assert (Hc : related_call M dead CS
                   (State (Apply ap) vars conts statics empty_heap rcd)
                   (State (Apply ap') vars'
                      ((w, wrapper_closure wps wmd k wargs vars' conts')
                       :: conts')
                      statics' empty_heap rcd)).
    { exists ap, ap', cs.
      cbn [state_expr state_vars state_conts state_statics state_heap
           state_reads].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ exact Hcall | ].
      split; [ exact Hvars | ].
      split.
      - rewrite Hk, Hw, resolve_head.
        destruct (resolve_rel2 M dead CS K conts conts' k Hconts)
          as [(HK & _) | (ksh & HK & Hck)]; rewrite Hlk in HK;
          [ discriminate HK | ].
        injection HK as <-. exact (cont_rel_wrap_u M dead CS _ _ _ _ _ _ _ _ _
                                     _ _ Hck Hok).
      - split; [ | split; [ exact Hst | repeat split ] ].
        intros extra Hextra.
        rewrite Hexn_eq, resolve_other
          by (intros Heq; exact (Hwe (eq_sym Heq))).
        exact (exn_rel K conts conts' vars vars' _ ap ap' Hconts Hvars Hexn
                 extra Hextra). }
    pose proof (call_answered _ _ Hc) as Hans. unfold call_answers in Hans.
    destruct (step (State (Apply ap) vars conts statics empty_heap rcd))
      as [s1 | k0 ws hp rd1 | ]; [ | destruct Hans | exact I ].
    destruct Hans as (t0 & n & t1 & Ht0 & Hn & Hrel).
    right. right. right. exists t1, n.
    split; [ | exact Hrel ].
    rewrite run_succ. unfold wrap. rewrite step_let_cont_non_recursive.
    unfold wrapper_closure in Ht0. rewrite run_succ, Ht0. exact Hn.
  - (* A call through wrappers of both runs. *)
    pose proof Hcall as (_ & _ & _ & _ & _ & _ & _ & _ & _ & Hexn_eq).
    unfold wrap. rewrite step_let_cont_non_recursive.
    right. left. eexists. split; [ rewrite step_let_cont_non_recursive;
                                   reflexivity | ].
    right. left. exists ap, ap', cs.
    cbn [state_expr state_vars state_conts state_statics state_heap
         state_reads].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ exact Hcall | ].
    split; [ exact Hvars | ].
    split.
    + rewrite Hwc, Hw, !resolve_head.
      destruct (resolve_rel2 M dead CS K conts conts' k Hconts)
        as [(HK & _) | (ksh & HK & Hck)]; rewrite Hlk in HK;
        [ discriminate HK | ].
      injection HK as <-. rewrite <- Hcmd.
      exact (cont_rel_wrap_both M dead CS _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hck
               Hok).
    + split; [ | split; [ exact Hst | repeat split ] ].
      intros extra Hextra.
      rewrite Hexn_eq, !resolve_other
        by (intros Heq; first [ exact (Hwe (eq_sym Heq))
                              | exact (Hwce (eq_sym Heq)) ]).
      exact (exn_rel K conts conts' vars vars' _ ap ap' Hconts Hvars Hexn
               extra Hextra).
Qed.

Lemma related2_answered (s t : state) : R s t -> answered R eq measure s t.
Proof.
  intros [Hm | [Hc | Hp]].
  - exact (main_answered s t Hm).
  - exact (call_answers_answered s t (call_answered s t Hc)).
  - exact (pending_answered s t Hp).
Qed.

(* Related terms, run on their own, refine each other. *)
Lemma uc_refines (p q : expr) :
  uc M dead CS [] p q ->
  refines (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Huc. apply (shrinking_refines R eq measure).
  - exact related2_answered.
  - intros vs vs' ->. reflexivity.
  - left. exists []. split; [ exact Huc | ].
    split; [ apply env_rel2_empty | split; [ constructor | ] ].
    split; [ apply statics_rel_empty | repeat split ].
Qed.

End Simulation.
