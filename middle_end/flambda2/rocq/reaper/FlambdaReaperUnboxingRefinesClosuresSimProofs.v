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

(* The relation of the proofs of FlambdaReaperUnboxingRefinesClosuresSpec.v
   between the unit rebuilt with the change of calling conventions alone
   (the first run) and the unit rebuilt with unboxing and changes of
   representation of sets of closures (the second run), and that it is a
   simulation ([rr_refines]).

   The names are of three classes. A plain name has one value in both
   runs. An unboxed name ([M]) is bound in the first run to a value whose
   fields, at their paths, the variables of its fields hold in the second
   (FlambdaReaperUnboxingRefinesProofs.v: [fields_hold], [env_rel]). A
   changed name ([CH]: a closure of a set whose representation changes)
   is bound in the first run to a closure at the slot [cr_current] of the
   representation, and in the second to the closure of the set made with
   the new slots ([crel]): at the new function slot, with the function
   declarations renamed, and holding at the new value slots of each field
   of the representation the value of the old slot, or, for a field
   unboxed in turn, the values at the paths of its fields. A changed name
   is in [M] with no field, so that it is not plain and [env_rel] asks
   nothing of it.

   The two terms are the same but where the second unboxes, as in
   FlambdaReaperUnboxingRefinesProofs.v ([rr_alloc], [rr_alias],
   [rr_load], [rr_read]), at the [Let] of a set whose representation
   changes ([rr_set]: the new slots, [new_slots]), and at a read of a
   value slot of a changed closure, which the second makes at the new
   slot ([rr_proj]), or at the new slots of the fields of an unboxed
   value ([rr_proj_unboxed]: [proj_lets]). Direct calls are the same in
   both ([rr_call]): full, with plain extra arguments for the exception
   continuation, and with a callee of the class of the [my_closure] of
   the code ([CS]: [callee_ok]), so that the code of both runs, related,
   runs with its [my_closure] related; but for the arguments given to the
   parameters that the second run unboxes. The code has the shapes of its
   parameters in the second run ([ci_params]: the same, or the variables
   of the fields of an unboxed one, at the arity [ci_params_arity]); with
   none unboxed, the code of both runs has the same parameters, of which
   it states the same, and the arguments are the same and plain;
   otherwise the parameters of the second are those of the first at
   those shapes, none changed, and what it states of them holds where
   what the first states does ([params_frame]), and the second run gives
   the variables of the fields of an unboxed name for an unboxed
   parameter ([args_frame]), so that the code enters with them bound to
   the fields of what the first gives ([call_code_entered_u]). The calls
   that return through wrappers or never return, and the calls whose code
   is not known, are of code none of whose parameters is unboxed
   ([call_ok_c], [ci_plain]). The exception continuation of a call with extra
   arguments is the frame of the handler with them, in both runs
   ([cont_rel_extra]). A call of a plain variable [f] whose code is not
   known is the same in both too ([rr_call_ind]): the callee holds the
   same value in both runs, so the code called has its [my_closure] and
   its parameters plain, as [ind_targets] asks of the code that the flow
   may bring to [f] ([RCH], [flow_ok] of each state of the first run,
   which [rr_refines_flow] takes); both runs make the same call of it, of
   the same shape ([closure_shape_rel]), into related code
   ([call_code_entered]). A call of known arity is a full call; one of
   unknown arity has one group of arguments, of code that has a
   parameter, so that it over-applies nothing, a partial application
   being made by the machine ([PGP.pa_bound]); or it may over-apply
   ([rr_call_over]), the code that the flow may bring to the closure of
   what [f] holds and the code that what is left of an over-application
   may call ([REM], which [flow_run] says of each state of the first run:
   PGP.step_rem_ok) having their [my_closure] plain: what is left of an
   over-application is related in the frames of the two runs
   ([cont_rel_over]), and both call the value they are given alike
   ([closure_call_answered]).
   Arguments of continuations are plain, so that the values of an exit
   are the same in both runs.

   The relation has dead continuations [D]: no case applies them or calls
   with them, and their frames need not be related ([conts_rel]). Code
   that never returns ([ci_never]) has its body related with its return
   continuation dead ([code_dead]). A call of it ([never_ok]: direct, or
   of a plain variable whose code is not known, every code that the flow
   may bring to it being such code, [ind_never], one of unknown arity a
   full call) may return, in the second run, to a handler [Invalid] bound
   before the call, and in the first to its continuation
   ([rr_call_never_r]) or to a wrapper ([rr_call_never_wraps]): the
   frames of the return continuation of the code are then any
   ([never_entered]). A handler that only the first run binds is dead
   ([rr_let_cont_never]). The free continuations of the second term are
   those of the first or dead, and it binds statics only where the first
   does ([rr_shape]); the dead continuations not free in it may change
   ([rr_dead_ext]).

   A call of [rr_call] may return through a wrapper of each run
   ([rr_call_wraps]): the two wrappers have other names, and parameters of
   other names ([W], the names that the rebuilds give the parameters of
   wrappers, which are no other names of the terms); given the same
   results they give the continuation of the call the same values
   ([wrap_ok]). The states are related by the terms ([related_main]); at
   the call, each run having bound its wrapper, the two frames of the
   wrappers related ([related_call], [cont_rel_wrap]); or within the
   wrappers, both applying the continuation, related, to the same values,
   the environments related at the names that are not of [W] only
   ([related_pending], [env_rel_cw]).

   Each step of the first run is answered by as many of the second as it
   makes [Let]s, none where it makes none, and a call that never returns
   by the [Let_cont] of the second and its call, so that the relation is
   a shrinking simulation (FlambdaReaperRebuildProofs.shrinking_refines).
   Other files read the relation only through the lemmas of the last
   section and [related_answered]. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaMapSpec FlambdaMapProofs
  FlambdaValueSpec FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaReaperGraphSpec
  FlambdaReaperRebuild FlambdaReaperRebuildProofs FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs.
From OxCamlFlambda2 Require FlambdaReaperPartialGroupsProofs
  FlambdaReaperUnboxingRefinesCallsSimProofs.
Import ListNotations.

Module PGP := FlambdaReaperPartialGroupsProofs.
Module CSP := FlambdaReaperUnboxingRefinesCallsSimProofs.

(* ======================================================================== *)
(* The relation                                                             *)
(* ======================================================================== *)

(* What a call of code needs of its callee: the class of the
   [my_closure] of the code (plain, or changed with that
   representation), the number of groups of its parameters, whether the
   code never returns, and what its parameters are in the second run
   (FlambdaReaperUnboxingRefinesCallsSimProofs.pshape: the same, or the
   variables of their fields), with the arity of the code in the second
   run, if some parameter is unboxed there. *)
Record code_info : Type := Code_info {
  ci_my_closure : option closure_repr;
  ci_nparams : nat;
  ci_never : bool;
  ci_params : list CSP.pshape;
  ci_params_arity : arity
}.

(* Shapes of parameters none of which is unboxed. *)
Definition plain_shapes (sh : list CSP.pshape) : bool :=
  forallb (fun o => match o with CSP.Psh_plain => true | _ => false end) sh.

(* A call of a closure whose code is not known enters code as the first
   run does: its [my_closure] and its parameters are plain. *)
Definition ci_plain (ci : code_info) : Prop :=
  ci_my_closure ci = None ∧ plain_shapes (ci_params ci) = true.

(* A call with the arguments [args] at the arity [aa]. *)
Definition with_args (ap : apply_expr) (args : list simple) (aa : arity)
    : apply_expr :=
  {| ap_callee := ap_callee ap;
     ap_call_kind := ap_call_kind ap;
     ap_args := args;
     ap_continuation := ap_continuation ap;
     ap_exn_continuation := ap_exn_continuation ap;
     ap_exn_extra_args := ap_exn_extra_args ap;
     ap_args_arity := aa;
     ap_return_arity := ap_return_arity ap |}.

(* The continuations that the body of code does not apply: its return
   continuation, if it never returns. *)
Definition code_dead (ci : code_info) (c : code) : list continuation :=
  if ci_never ci then [code_return_continuation c] else [].

(* A read of a value slot, and from what. *)
Definition slot_read (d : named) : option (value_slot * simple) :=
  match d with
  | Project_value_slot w s | Project_value_slot_from _ w s => Some (w, s)
  | _ => None
  end.

(* The [Let]s that read the new slots [pairs] of a changed closure [y] at
   the function slot [cs], the first the innermost
   ([load_field_from_value_which_is_being_unboxed]). *)
Definition proj_lets (cs : function_slot) (y : simple)
    (pairs : list (variable * variable * flambda_kind)) (body : expr)
    : expr :=
  List.fold_left
    (fun body '(x, slot, kind) =>
       Let x kind (Project_value_slot_from cs slot y) body)
    pairs body.

(* The wrappers of a call in the two runs, of parameters [cps] and [wps],
   which give the continuation of the call the arguments [cargs] and
   [wargs]: given what the callee returns, the second accepts it where the
   first does, and gives the continuation what the first gives it. *)
Definition wrap_ok (cps : list variable) (cmd : handler_metadata)
    (cargs : list simple) (wps : list variable) (wmd : handler_metadata)
    (wargs : list simple) : Prop :=
  hm_params_kinds wmd = hm_params_kinds cmd
  ∧ (∀ (S : static_env) (vs : list value),
       values_of_subkinds S vs (hm_params_subkinds cmd) = true ->
       values_of_subkinds S vs (hm_params_subkinds wmd) = true)
  ∧ ∀ (vs : list value) (cv wv hv : env),
      bind_params cps vs cv = Some hv ->
      ∃ hv' : env,
        bind_params wps vs wv = Some hv'
        ∧ ∀ ws : list value,
            eval_simples hv cargs = Some ws ->
            eval_simples hv' wargs = Some ws.

(* The body of a wrapper, which gives [k] the arguments [args]. *)
Definition wrapper_body (k : continuation) (args : list simple) : expr :=
  Apply_cont
    {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Section Related.

Variable M : list (node * unboxed_fields).
Variable CH : variable -> option closure_repr.
Variable CS : code_id -> option code_info.
(* The code that the flow may bring to a variable [f]: [RCH true f] that of
   a closure that [f] holds, [RCH false f] that of the closure that a call
   of what [f] holds calls (of a partial application, its closure). *)
Variable RCH : bool -> variable -> code_id -> Prop.
(* The names that the two rebuilds give the parameters of the wrappers of
   calls, which are no other names of the terms: within a wrapper, the
   environments are related at the other names only ([env_rel_cw]). *)
Variable W : variable -> Prop.
(* The code that what is left of an over-application may call, which
   [flow_run] says of each state of the first run. *)
Variable REM : code_id -> Prop.

(* The new value slots of the field [fu] of a representation, for a set
   whose old value slots are [vs]: the new slot of a field not unboxed
   holds what the old one did; those of a field unboxed in turn hold the
   variables of the fields of the unboxed name that the old one held. *)
Definition new_slot (vs : list (value_slot * simple))
    (fu : field * unboxed_field) : list (value_slot * simple) :=
  match fu with
  | (Field_value_slot w, u) =>
    match lookup w vs with
    | Some a =>
      match u with
      | Not_unboxed s => [(s, a)]
      | Unboxed _ =>
        match a with
        | Var b =>
          match fields_of M b with
          | Some fb =>
            match unboxed_field_pairs Value u (Unboxed fb) with
            | Some pairs => map (fun '(s, x, _) => (s, Var x)) pairs
            | None => []
            end
          | None => []
          end
        | Symbol _ | Const _ => []
        end
      end
    | None => []
    end
  | _ => []
  end.

Definition new_slots (r : closure_repr) (vs : list (value_slot * simple))
    : list (value_slot * simple) :=
  flat_map (new_slot vs) (cr_fields r).

(* A field of a representation, for a set whose value slots are [vs]: a
   value slot of the set, holding a plain simple or, if the field is
   unboxed in turn, a name unboxed alike. *)
Definition set_field_ok (vs : list (value_slot * simple))
    (fu : field * unboxed_field) : Prop :=
  match fu with
  | (Field_value_slot w, u) =>
    ∃ a : simple,
      lookup w vs = Some a
      ∧ match u with
        | Not_unboxed _ => plain_simple M a = true
        | Unboxed _ =>
          ∃ (b : variable) (fb : unboxed_fields)
            (pairs : list (variable * variable * flambda_kind)),
            a = Var b ∧ CH b = None ∧ fields_of M b = Some fb
            ∧ unboxed_field_pairs Value u (Unboxed fb) = Some pairs
        end
  | _ => False
  end.

Definition callee_ok (ci : code_info) (s : simple) : Prop :=
  match ci_my_closure ci with
  | None => plain_simple M s = true
  | Some r => ∃ f : variable, s = Var f ∧ CH f = Some r
  end.

Definition my_closure_ok (ci : code_info) (z : variable) : Prop :=
  match ci_my_closure ci with
  | None => plain_variable M z = true
  | Some r => CH z = Some r
  end.

(* The parameters of the code of the two runs: the same, plain, if none
   is unboxed in the second; otherwise those of the second are those of
   the first at the shapes of [ci], none changed
   (FlambdaReaperUnboxingRefinesCallsSimProofs.params_rel2), at the arity
   of [ci], whose kinds are those of the first at those shapes, and what
   the second states of them holds where what the first states does. *)
Definition params_frame (ci : code_info) (c c' : code) : Prop :=
  if plain_shapes (ci_params ci) then
    code_params c' = code_params c
    ∧ code_params_arity c' = code_params_arity c
    ∧ forallb (plain_variable M) (code_params c) = true
    ∧ code_params_subkinds c' = code_params_subkinds c
  else
    CSP.no_dead (ci_params ci)
    ∧ (∀ p : variable, p ∈ code_params c -> CH p = None)
    ∧ (∃ pk pk' : list (variable * flambda_kind),
         code_params c = map fst pk ∧ code_params c' = map fst pk'
         ∧ CSP.params_rel2 M [] (ci_params ci) pk pk')
    ∧ code_params_arity c' = ci_params_arity ci
    ∧ length (ci_params_arity ci) = ci_nparams ci
    ∧ CSP.shape_kinds (ci_params ci) (unarize (code_params_arity c))
      = unarize (ci_params_arity ci)
    ∧ CSP.sks_follow2 (ci_params ci) (code_params_subkinds c)
        (code_params_subkinds c').

(* What the code of the two runs shares but for its body. *)
Definition code_frame (id : code_id) (c c' : code) (ci : code_info) : Prop :=
  CS id = Some ci
  ∧ params_frame ci c c'
  ∧ code_my_closure c' = code_my_closure c
  ∧ code_return_continuation c' = code_return_continuation c
  ∧ code_exn_continuation c' = code_exn_continuation c
  ∧ code_result_arity c' = code_result_arity c
  ∧ code_is_tupled c = false ∧ code_is_tupled c' = false
  ∧ length (code_params_arity c) = ci_nparams ci
  ∧ my_closure_ok ci (code_my_closure c).

(* The arguments [args'], at the arity [aa'], that the second run gives
   code of class [ci] for the arguments of the call [ap] of the first: the
   same, plain, if no parameter is unboxed in the second; otherwise the
   variables of the fields of those given to an unboxed parameter
   (FlambdaReaperUnboxingRefinesCallsSimProofs.args_rel2), at the arity
   of [ci]. *)
Definition args_frame (ci : code_info) (ap : apply_expr) (args' : list simple)
    (aa' : arity) : Prop :=
  if plain_shapes (ci_params ci) then
    args' = ap_args ap ∧ aa' = ap_args_arity ap
    ∧ forallb (plain_simple M) (ap_args ap) = true
  else
    CSP.args_rel2 M [] (ci_params ci) (ap_args ap) args'
    ∧ aa' = ci_params_arity ci
    ∧ CSP.shape_kinds (ci_params ci) (unarize (ap_args_arity ap))
      = unarize (ci_params_arity ci).

(* A call of the code [id] whose class is [ci]: as [rr_call]. *)
Definition call_ok_c (ap : apply_expr) (id : code_id) (ci : code_info)
    : Prop :=
  ap_call_kind ap = Direct id ∧ CS id = Some ci ∧ callee_ok ci (ap_callee ap)
  ∧ forallb (plain_simple M) (ap_args ap) = true
  ∧ forallb (plain_simple M) (exn_extra_simples ap) = true
  ∧ length (ap_args_arity ap) = ci_nparams ci
  ∧ plain_shapes (ci_params ci) = true.

(* A call [ap] of [f] whose code is not known: of known arity, of code
   whose [my_closure] is plain where the flow may bring it to [f]; or of
   unknown arity, with one group of arguments, of code whose [my_closure]
   is plain and that has a parameter where the flow may bring it to the
   closure of what [f] holds. *)
Definition ind_targets (ap : apply_expr) (f : variable) : Prop :=
  match ap_call_kind ap with
  | Direct _ => False
  | Indirect_known_arity _ =>
    ∀ (id : code_id) (ci : code_info),
      RCH true f id -> CS id = Some ci -> ci_plain ci
  | Indirect_unknown_arity =>
    length (ap_args_arity ap) = 1
    ∧ ∀ (id : code_id) (ci : code_info),
        RCH false f id -> CS id = Some ci ->
        ci_plain ci ∧ ci_nparams ci <> 0
  end.

(* A call [ap] of [f] whose code is not known, of code that never
   returns: every code that the flow may bring to [f] never returns and
   has its [my_closure] plain; for a call of unknown arity, which has one
   group of arguments, it has one parameter, so that the call is a full
   call. *)
Definition ind_never (ap : apply_expr) (f : variable) : Prop :=
  match ap_call_kind ap with
  | Direct _ => False
  | Indirect_known_arity _ =>
    ∀ (id : code_id) (ci : code_info),
      RCH true f id -> CS id = Some ci ->
      ci_plain ci ∧ ci_never ci = true
  | Indirect_unknown_arity =>
    length (ap_args_arity ap) = 1
    ∧ ∀ (id : code_id) (ci : code_info),
        RCH false f id -> CS id = Some ci ->
        ci_plain ci ∧ ci_never ci = true ∧ ci_nparams ci = 1
  end.

(* A call of code that never returns: of the code [id] of class [ci], as
   [rr_call]; or of a plain variable whose code is not known, as
   [ind_never] says. *)
Definition never_ok (ap : apply_expr) : Prop :=
  (∃ (id : code_id) (ci : code_info),
     call_ok_c ap id ci ∧ ci_never ci = true)
  ∨ ∃ f : variable,
      ap_callee ap = Var f ∧ plain_variable M f = true
      ∧ forallb (plain_simple M) (ap_args ap) = true
      ∧ forallb (plain_simple M) (exn_extra_simples ap) = true
      ∧ ind_never ap f.

(* The terms related, the continuations [D] dead: no case applies them, or
   calls with them, and their frames are not related ([conts_rel]). *)
Inductive rr : list continuation -> expr -> expr -> Prop :=
  | rr_let (D : list continuation)
      (x : variable) (kind : flambda_kind) (d : named)
      (body body' : expr) :
    plain_variable M x = true
    -> FlambdaReaperUnboxingRefinesProofs.named_plain M d = true ->
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    rr D body body' -> rr D (Let x kind d body) (Let x kind d body')
  | rr_plain_set (D : list continuation)
      (x : variable) (kind : flambda_kind) (slot : function_slot)
      (decls : list (function_slot * code_id))
      (vs : list (value_slot * simple)) (body body' : expr) :
    plain_variable M x = true ->
    forallb (fun ws => plain_simple M (snd ws)) vs = true ->
    rr D body body' ->
    rr D (Let x kind (Closure_of_set slot decls vs) body)
      (Let x kind (Closure_of_set slot decls vs) body')
  | rr_alloc (D : list continuation)
      (x : variable) (d : named) (fields : unboxed_fields)
      (ls : list (list (variable * flambda_kind * simple)))
      (body body' : expr) :
    fields_of M x = Some fields -> CH x = None ->
    among x (field_variables M) = false ->
    Forall2 (alloc_entry M x d) fields ls ->
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    rr D body body' -> rr D (Let x Value d body) (lets (concat ls) body')
  | rr_alias (D : list continuation)
      (x y : variable) (fields fields_y : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind)) (body body' : expr) :
    fields_of M x = Some fields -> CH x = None ->
    among x (field_variables M) = false ->
    fields_of M y = Some fields_y -> x <> y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs ->
    rr D body body' ->
    rr D (Let x Value (Simple (Var y)) body) (lets (pairs_lets pairs) body')
  | rr_load (D : list continuation)
      (x : variable) (d : named) (f : field) (s : simple) (y : variable)
      (fields fields_y nested : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind)) (body body' : expr) :
    fields_of M x = Some fields -> CH x = None ->
    among x (field_variables M) = false ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields_y -> x <> y ->
    (f, Unboxed nested) ∈ fields_y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed nested)
    = Some pairs ->
    rr D body body' -> rr D (Let x Value d body) (lets (pairs_lets pairs) body')
  | rr_read (D : list continuation)
      (x : variable) (kind : flambda_kind) (d : named) (f : field)
      (s : simple) (y v : variable) (fields : unboxed_fields)
      (body body' : expr) :
    plain_variable M x = true -> read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields -> (f, Not_unboxed v) ∈ fields ->
    rr D body body' ->
    rr D (Let x kind d body) (Let x kind (Simple (Var v)) body')
  | rr_set (D : list continuation)
      (x : variable) (r : closure_repr)
      (decls : list (function_slot * code_id))
      (vs : list (value_slot * simple)) (body body' : expr) :
    CH x = Some r ->
    lookup (current_slot r) (rename_function_decls r decls)
    = lookup (cr_current r) decls ->
    Forall (set_field_ok vs) (cr_fields r) ->
    NoDup (map fst (new_slots r vs)) ->
    rr D body body' ->
    rr D (Let x Value (Closure_of_set (cr_current r) decls vs) body)
      (Let x Value
         (Closure_of_set (current_slot r) (rename_function_decls r decls)
            (new_slots r vs))
         body')
  | rr_proj (D : list continuation)
      (x : variable) (kind : flambda_kind) (d : named) (w : value_slot)
      (s : simple) (y : variable) (r : closure_repr) (sl : value_slot)
      (body body' : expr) :
    plain_variable M x = true -> slot_read d = Some (w, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    CH y = Some r -> (Field_value_slot w, Not_unboxed sl) ∈ cr_fields r ->
    rr D body body' ->
    rr D (Let x kind d body)
      (Let x kind (Project_value_slot_from (current_slot r) sl (Var y))
         body')
  | rr_proj_unboxed (D : list continuation)
      (x : variable) (d : named) (w : value_slot) (s : simple)
      (y : variable) (r : closure_repr) (fields nested : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind)) (body body' : expr) :
    fields_of M x = Some fields -> CH x = None ->
    among x (field_variables M) = false ->
    slot_read d = Some (w, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    CH y = Some r -> (Field_value_slot w, Unboxed nested) ∈ cr_fields r ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed nested)
    = Some pairs ->
    rr D body body' ->
    rr D (Let x Value d body) (proj_lets (current_slot r) (Var y) pairs body')
  | rr_static (D : list continuation) (bs bs' : list static_binding)
      (body body' : expr) :
    rr_bindings bs bs' -> rr D body body' ->
    rr D (Let_static bs body) (Let_static bs' body')
  | rr_let_cont (D : list continuation) (k : continuation)
      (ps : list variable) (md : handler_metadata) (h h' body body' : expr) :
    forallb (plain_variable M) ps = true -> rr D h h' ->
    rr (without [k] D) body body' ->
    rr D (Let_cont_non_recursive k (Handler ps md h) body)
      (Let_cont_non_recursive k (Handler ps md h') body')
  | rr_let_cont_rec (D : list continuation)
      (hs hs' : list (continuation * continuation_handler))
      (body body' : expr) :
    rr_handlers (without (map fst hs) D) hs hs' ->
    rr (without (map fst hs) D) body body' ->
    rr D (Let_cont_recursive hs body) (Let_cont_recursive hs' body')
  | rr_apply_cont (D : list continuation) (ac : apply_cont) :
    forallb (plain_simple M) (ac_args ac) = true -> ac_continuation ac ∉ D ->
    rr D (Apply_cont ac) (Apply_cont ac)
  | rr_switch (D : list continuation) (scrutinee : simple)
      (arms : list (Z * apply_cont)) :
    plain_simple M scrutinee = true ->
    forallb (fun arm => forallb (plain_simple M) (ac_args (snd arm))) arms
    = true ->
    (∀ arm : Z * apply_cont, arm ∈ arms -> ac_continuation (snd arm) ∉ D) ->
    rr D (Switch scrutinee arms) (Switch scrutinee arms)
  | rr_invalid (D : list continuation) : rr D Invalid Invalid
  | rr_call (D : list continuation) (ap : apply_expr) (id : code_id)
      (ci : code_info) (args' : list simple) (aa' : arity) :
    ap_call_kind ap = Direct id -> CS id = Some ci ->
    callee_ok ci (ap_callee ap) ->
    args_frame ci ap args' aa' ->
    forallb (plain_simple M) (exn_extra_simples ap) = true ->
    length (ap_args_arity ap) = ci_nparams ci ->
    ap_continuation ap ∉ D -> ap_exn_continuation ap ∉ D ->
    rr D (Apply ap) (Apply (with_args ap args' aa'))
  (* A call of a plain variable whose code is not known. *)
  | rr_call_ind (D : list continuation) (ap : apply_expr) (f : variable) :
    ap_callee ap = Var f -> plain_variable M f = true -> ind_targets ap f ->
    forallb (plain_simple M) (ap_args ap) = true ->
    forallb (plain_simple M) (exn_extra_simples ap) = true ->
    ap_continuation ap ∉ D -> ap_exn_continuation ap ∉ D ->
    rr D (Apply ap) (Apply ap)
  (* A call of code that never returns, each run returning anywhere: the
     second run's continuation is dead. *)
  | rr_call_never (D : list continuation) (ap : apply_expr) (k : continuation) :
    never_ok ap -> k ∈ D -> ap_exn_continuation ap ∉ D ->
    rr D (Apply ap) (Apply (with_continuation ap k))
  (* The same, returning in the second run to a handler [Invalid]. *)
  | rr_call_never_r (D : list continuation) (ap : apply_expr)
      (w : continuation) (wps : list variable) (wmd : handler_metadata) :
    never_ok ap ->
    w <> ap_exn_continuation ap -> ap_exn_continuation ap ∉ D ->
    rr D (Apply ap)
      (Let_cont_non_recursive w (Handler wps wmd Invalid)
         (Apply (with_continuation ap w)))
  (* The same, returning in the first run to a wrapper. *)
  | rr_call_never_wraps (D : list continuation) (ap : apply_expr)
      (wc : continuation) (cps : list variable) (cmd : handler_metadata)
      (hb : expr) (w : continuation) (wps : list variable)
      (wmd : handler_metadata) :
    never_ok ap ->
    wc <> ap_exn_continuation ap -> w <> ap_exn_continuation ap ->
    ap_exn_continuation ap ∉ D ->
    rr D (Let_cont_non_recursive wc (Handler cps cmd hb) (Apply ap))
      (Let_cont_non_recursive w (Handler wps wmd Invalid)
         (Apply (with_continuation ap w)))
  (* A handler that only the first run binds, which is dead. *)
  | rr_let_cont_never (D : list continuation) (k : continuation)
      (h : continuation_handler) (body body' : expr) :
    k ∉ free_continuations_of_expr body' -> rr (k :: D) body body' ->
    rr D (Let_cont_non_recursive k h body) body'
  (* A call of [rr_call] that returns through a wrapper in each run, of
     parameters of [W]: the wrappers give its continuation [k] what the
     callee returns alike. *)
  | rr_call_wraps (D : list continuation) (ap : apply_expr) (id : code_id)
      (ci : code_info) (k wc : continuation) (cps : list variable)
      (cmd : handler_metadata) (cargs : list simple) (w : continuation)
      (wps : list variable) (wmd : handler_metadata) (wargs : list simple) :
    call_ok_c ap id ci ->
    wc <> ap_exn_continuation ap -> w <> ap_exn_continuation ap ->
    k ∉ D -> ap_exn_continuation ap ∉ D ->
    (∀ z : variable, z ∈ cps ++ wps -> W z) ->
    wrap_ok cps cmd cargs wps wmd wargs ->
    rr D
      (Let_cont_non_recursive wc (Handler cps cmd (wrapper_body k cargs))
         (Apply (with_continuation ap wc)))
      (Let_cont_non_recursive w (Handler wps wmd (wrapper_body k wargs))
         (Apply (with_continuation ap w)))
  (* A call of unknown arity of a plain variable [f] whose code is not
     known, which may over-apply: the code that the flow may bring to the
     closure of what [f] holds, and the code that what is left of an
     over-application may call, have their [my_closure] plain. *)
  | rr_call_over (D : list continuation) (ap : apply_expr) (f : variable) :
    ap_callee ap = Var f -> plain_variable M f = true ->
    ap_call_kind ap = Indirect_unknown_arity ->
    (∀ (id : code_id) (ci : code_info),
       RCH false f id ∨ REM id -> CS id = Some ci -> ci_plain ci)
    -> forallb (plain_simple M) (ap_args ap) = true ->
    forallb (plain_simple M) (exn_extra_simples ap) = true ->
    ap_continuation ap ∉ D -> ap_exn_continuation ap ∉ D ->
    rr D (Apply ap) (Apply ap)
with rr_handlers : list continuation
    -> list (continuation * continuation_handler)
    -> list (continuation * continuation_handler) -> Prop :=
  | rr_handlers_nil (D : list continuation) : rr_handlers D [] []
  | rr_handlers_cons (D : list continuation) (k : continuation)
      (ps : list variable) (md : handler_metadata) (h h' : expr)
      (hs hs' : list (continuation * continuation_handler)) :
    forallb (plain_variable M) ps = true -> rr D h h' ->
    rr_handlers D hs hs' ->
    rr_handlers D ((k, Handler ps md h) :: hs) ((k, Handler ps md h') :: hs')
with rr_bindings : list static_binding -> list static_binding -> Prop :=
  | rr_bindings_nil : rr_bindings [] []
  | rr_bindings_symbol (b : static_binding) (bs bs' : list static_binding) :
    FlambdaReaperUnboxingRefinesProofs.local_binding M b = true ->
    rr_bindings bs bs' -> rr_bindings (b :: bs) (b :: bs')
  | rr_bindings_deleted (id : code_id) (bs bs' : list static_binding) :
    rr_bindings bs bs' ->
    rr_bindings (Bind_deleted_code id :: bs) (Bind_deleted_code id :: bs')
  | rr_bindings_code (id : code_id) (c c' : code) (ci : code_info)
      (bs bs' : list static_binding) :
    code_frame id c c' ci ->
    rr (code_dead ci c) (code_body c) (code_body c') ->
    rr_bindings bs bs' ->
    rr_bindings (Bind_code id c :: bs) (Bind_code id c' :: bs').

(* The values of the new slots of a field [u] of a representation, the
   old slot holding [v]. *)
Definition slot_holds (vs' : list (value_slot * value)) (u : unboxed_field)
    (v : value) : Prop :=
  match u with
  | Not_unboxed s => lookup s vs' = Some v
  | Unboxed fields => fields_hold vs' fields v
  end.

(* A closure of the first run and the closure of the second, of a set
   whose representation is [r]. *)
Definition crel (r : closure_repr) (v v' : value) : Prop :=
  ∃ (decls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)),
    v = Value_closure_of_set (cr_current r) decls vs
    ∧ v' = Value_closure_of_set (current_slot r) (rename_function_decls r decls)
             vs'
    ∧ lookup (current_slot r) (rename_function_decls r decls)
      = lookup (cr_current r) decls
    ∧ ∀ (w : value_slot) (u : unboxed_field),
        (Field_value_slot w, u) ∈ cr_fields r ->
        ∃ v0 : value, lookup w vs = Some v0 ∧ slot_holds vs' u v0.

(* [crel], with the value slots of the closure of the second run exactly
   the new slots of the representation, in their order, each once, as the
   rebuild makes them (FlambdaReaperUnboxing.changed_value_slots); the
   fields of the representation are value slots, and the closure of the
   first run is at a function slot of its set. *)
Definition crel_exact (r : closure_repr) (v v' : value) : Prop :=
  crel r v v'
  ∧ ∃ (decls : list (function_slot * code_id))
      (vs vs' : list (value_slot * value)),
      v = Value_closure_of_set (cr_current r) decls vs
      ∧ v'
        = Value_closure_of_set (current_slot r)
            (rename_function_decls r decls) vs'
      ∧ lookup (cr_current r) decls <> None
      ∧ map fst vs' = map fst (fields_leaves (cr_fields r))
      ∧ NoDup (map fst vs')
      ∧ ∀ fu : field * unboxed_field,
          fu ∈ cr_fields r ->
          ∃ (w : value_slot) (u : unboxed_field),
            fu = (Field_value_slot w, u).

Definition env_rel_c (vars vars' : env) : Prop :=
  env_rel M vars vars'
  ∧ ∀ (x : variable) (r : closure_repr) (w : value),
      CH x = Some r -> lookup x vars = Some w ->
      ∃ w' : value, lookup x vars' = Some w' ∧ crel_exact r w w'.

(* [env_rel_c] at the names that are not of [W]: the plain names, and
   the changed ones. *)
Definition env_rel_cw (vars vars' : env) : Prop :=
  (∀ z : variable,
     ~ W z -> plain_variable M z = true -> lookup z vars = lookup z vars')
  ∧ ∀ (x : variable) (r : closure_repr) (w : value),
      ~ W x -> CH x = Some r -> lookup x vars = Some w ->
      ∃ w' : value, lookup x vars' = Some w' ∧ crel_exact r w w'.

Definition code_rel (id : code_id) (c c' : code) : Prop :=
  ∃ ci : code_info,
    code_frame id c c' ci ∧ rr (code_dead ci c) (code_body c) (code_body c').

Definition statics_rel (S S' : static_env) : Prop :=
  static_values S' = static_values S
  ∧ map fst (static_code S') = map fst (static_code S)
  ∧ ∀ (id : code_id) (c : code),
      lookup id (static_code S) = Some c ->
      ∃ c' : code, lookup id (static_code S') = Some c' ∧ code_rel id c c'.

(* Frames related. A handler holds frames related at the continuations
   not in its [D], and any frames at those in [D], of both runs or of
   one. *)
Inductive cont_rel : cont_value -> cont_value -> Prop :=
  | cont_rel_closure (D : list continuation) (ps : list variable)
      (md : handler_metadata) (h h' : expr) (vars vars' : env)
      (conts conts' : cont_env)
      (rec rec' : list (continuation * continuation_handler)) :
    forallb (plain_variable M) ps = true ->
    rr (without (map fst rec) D) h h' -> env_rel_c vars vars' ->
    (∀ k : continuation,
       k ∉ D -> cont_rel (resolve k conts) (resolve k conts')) ->
    rr_handlers (without (map fst rec) D) rec rec' ->
    cont_rel (Closure (Handler ps md h) vars conts rec)
      (Closure (Handler ps md h') vars' conts' rec')
  | cont_rel_outer (k : continuation) : cont_rel (Outer k) (Outer k)
  | cont_rel_extra (c c' : cont_value) (extra : list value) :
    cont_rel c c' -> cont_rel (Extra_args c extra) (Extra_args c' extra)
  (* The wrappers of a call, made at related states. *)
  | cont_rel_wrap (cps : list variable) (cmd : handler_metadata)
      (k : continuation) (cargs : list simple) (cvars : env)
      (cconts : cont_env) (wps : list variable) (wmd : handler_metadata)
      (wargs : list simple) (wvars : env) (wconts : cont_env) :
    cont_rel (resolve k cconts) (resolve k wconts) -> env_rel_c cvars wvars ->
    (∀ z : variable, z ∈ cps ++ wps -> W z) ->
    wrap_ok cps cmd cargs wps wmd wargs ->
    cont_rel (Closure (Handler cps cmd (wrapper_body k cargs)) cvars cconts [])
      (Closure (Handler wps wmd (wrapper_body k wargs)) wvars wconts [])
  (* What is left of an over-application in both runs, the code of [REM]
     having its [my_closure] plain. *)
  | cont_rel_over (remaining : list value) (ra rta : arity)
      (c c' e e' : cont_value) :
    cont_rel c c' -> cont_rel e e' ->
    (∀ (id : code_id) (ci : code_info),
       REM id -> CS id = Some ci -> ci_plain ci) ->
    cont_rel (Over_application remaining ra rta c e)
      (Over_application remaining ra rta c' e').

Definition conts_rel (D : list continuation) (conts conts' : cont_env)
    : Prop :=
  ∀ k : continuation, k ∉ D -> cont_rel (resolve k conts) (resolve k conts').

(* The states related: by the terms ([related_main]); both at a call that
   returns through a wrapper of each, the wrappers related
   ([related_call]); or both within a wrapper, applying a continuation
   related to the same values ([related_pending]). *)
Definition related_main (s t : state) : Prop :=
  ∃ D : list continuation,
  rr D (state_expr s) (state_expr t)
  ∧ env_rel_c (state_vars s) (state_vars t)
  ∧ conts_rel D (state_conts s) (state_conts t)
  ∧ statics_rel (state_statics s) (state_statics t)
  ∧ state_heap s = empty_heap ∧ state_heap t = empty_heap
  ∧ state_reads s = state_reads t.

Definition related_call (s t : state) : Prop :=
  ∃ (ap : apply_expr) (k2 : continuation) (id : code_id) (ci : code_info),
    state_expr s = Apply ap ∧ state_expr t = Apply (with_continuation ap k2)
    ∧ call_ok_c ap id ci
    ∧ env_rel_c (state_vars s) (state_vars t)
    ∧ cont_rel (resolve (ap_continuation ap) (state_conts s))
        (resolve k2 (state_conts t))
    ∧ cont_rel (resolve (ap_exn_continuation ap) (state_conts s))
        (resolve (ap_exn_continuation ap) (state_conts t))
    ∧ statics_rel (state_statics s) (state_statics t)
    ∧ state_heap s = empty_heap ∧ state_heap t = empty_heap
    ∧ state_reads s = state_reads t.

Definition related_pending (s t : state) : Prop :=
  ∃ ac ac' : apply_cont,
    state_expr s = Apply_cont ac ∧ state_expr t = Apply_cont ac'
    ∧ ac_continuation ac' = ac_continuation ac
    ∧ cont_rel (resolve (ac_continuation ac) (state_conts s))
        (resolve (ac_continuation ac) (state_conts t))
    ∧ (∀ vs : list value,
         eval_simples (state_vars s) (ac_args ac) = Some vs ->
         eval_simples (state_vars t) (ac_args ac') = Some vs)
    ∧ env_rel_cw (state_vars s) (state_vars t)
    ∧ statics_rel (state_statics s) (state_statics t)
    ∧ state_heap s = empty_heap ∧ state_heap t = empty_heap
    ∧ state_reads s = state_reads t.

Definition related (s t : state) : Prop :=
  related_main s t ∨ related_call s t ∨ related_pending s t.

End Related.

Scheme rr_mut := Induction for rr Sort Prop
  with rr_handlers_mut := Induction for rr_handlers Sort Prop
  with rr_bindings_mut := Induction for rr_bindings Sort Prop.
Combined Scheme rr_comb from rr_mut, rr_handlers_mut, rr_bindings_mut.

(* ======================================================================== *)
(* The new slots of a set, and the reads of them                            *)
(* ======================================================================== *)

Lemma eval_value_slots_lookup (vars : env) (l : list (value_slot * simple))
    (vs : list (value_slot * value)) (w : value_slot) (a : simple) :
  eval_value_slots vars l = Some vs -> lookup w l = Some a ->
  ∃ v : value, eval_simple vars a = Some v ∧ lookup w vs = Some v.
Proof.
  revert vs. induction l as [ | [w0 s0] l IH]; intros vs Hl Ha;
    [ discriminate Ha | ].
  cbn [eval_value_slots] in Hl.
  destruct (eval_simple vars s0) as [v0 | ] eqn:Hs0; [ | discriminate Hl ].
  destruct (eval_value_slots vars l) as [vs0 | ] eqn:Hvs0;
    [ | discriminate Hl ].
  cbn in Hl. injection Hl as <-. rewrite lookup_cons in Ha.
  rewrite lookup_cons.
  destruct (Pos.eqb w w0).
  - injection Ha as <-. exists v0. split; [ exact Hs0 | reflexivity ].
  - exact (IH vs0 eq_refl Ha).
Qed.

Lemma eval_value_slots_some (vars : env) (l : list (value_slot * simple)) :
  (∀ (s : value_slot) (a : simple),
     (s, a) ∈ l -> ∃ v : value, eval_simple vars a = Some v) ->
  ∃ vs : list (value_slot * value), eval_value_slots vars l = Some vs.
Proof.
  induction l as [ | [w0 s0] l IH]; intros Hall; [ exists []; reflexivity | ].
  destruct (Hall w0 s0 (or_introl eq_refl)) as (v0 & Hv0).
  destruct IH as (vs & Hvs);
    [ intros s a Hin; exact (Hall s a (or_intror Hin)) | ].
  exists ((w0, v0) :: vs). cbn [eval_value_slots]. rewrite Hv0, Hvs.
  reflexivity.
Qed.

Lemma eval_value_slots_in (vars : env) (l : list (value_slot * simple))
    (vs : list (value_slot * value)) (s : value_slot) (a : simple)
    (v : value) :
  eval_value_slots vars l = Some vs -> NoDup (map fst l) -> (s, a) ∈ l ->
  eval_simple vars a = Some v -> lookup s vs = Some v.
Proof.
  revert vs. induction l as [ | [w0 s0] l IH]; intros vs Hl Hnd Hin Ha;
    [ destruct Hin | ].
  cbn [eval_value_slots] in Hl.
  destruct (eval_simple vars s0) as [v0 | ] eqn:Hs0; [ | discriminate Hl ].
  destruct (eval_value_slots vars l) as [vs0 | ] eqn:Hvs0;
    [ | discriminate Hl ].
  cbn in Hl. injection Hl as <-. cbn [map fst] in Hnd.
  inversion Hnd as [ | w1 l1 Hw0 Hnd' [Heq1 Heq2]]; subst.
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. rewrite Ha in Hs0. injection Hs0 as ->.
    apply lookup_head.
  - rewrite lookup_other; [ exact (IH vs0 eq_refl Hnd' Hin Ha) | ].
    intros <-. apply Hw0. apply in_map_iff. exists (s, a).
    split; [ reflexivity | exact Hin ].
Qed.


(* A read of a value slot of a changed closure: in the first run, the old
   slot of the closure; and the new slots of the second run are read at
   the new function slot. *)
Lemma slot_read_crel (S S' : static_env) (vars : env) (d : named)
    (w : value_slot) (s : simple) (y : variable) (r : closure_repr)
    (v v' r0 : value) :
  slot_read d = Some (w, s) ->
  (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
  lookup y vars = Some v -> crel r v v' -> eval_named S vars d = Some r0 ->
  ∃ vs' : list (value_slot * value),
    (∀ u : unboxed_field,
       (Field_value_slot w, u) ∈ cr_fields r -> slot_holds vs' u r0)
    ∧ ∀ sl : value_slot,
        project_value_slot S' (current_slot r) sl v' = lookup sl vs'.
Proof.
  intros Hd Hs Hy (decls & vs & vs' & -> & -> & Hdecls & Hslots) Hev.
  assert (Hread : lookup (cr_current r) decls <> None
                  ∧ lookup w vs = Some r0).
  { destruct d as [ | | | | w0 s0 | fs w0 s0 | ]; try discriminate Hd;
      cbn [slot_read] in Hd; injection Hd as <- <-;
      destruct Hs as [-> | (k & ->)]; cbn [eval_named eval_simple] in Hev;
      try (rewrite Hy in Hev); cbn in Hev; try discriminate Hev.
    - destruct (lookup (cr_current r) decls) as [code | ] eqn:Hc;
        cbn in Hev; [ | discriminate Hev ].
      split; [ discriminate | exact Hev ].
    - unfold project_value_slot in Hev. cbn in Hev.
      destruct (lookup (cr_current r) decls) as [code | ] eqn:Hc;
        cbn in Hev; [ | discriminate Hev ].
      destruct (Pos.eqb (cr_current r) fs); [ | discriminate Hev ].
      split; [ discriminate | exact Hev ]. }
  destruct Hread as [Hcur Hw].
  exists vs'. split.
  - intros u Hin. destruct (Hslots w u Hin) as (v0 & Hv0 & Hh).
    rewrite Hw in Hv0. injection Hv0 as <-. exact Hh.
  - intros sl. unfold project_value_slot. cbn. rewrite Hdecls.
    destruct (lookup (cr_current r) decls) as [code | ];
      [ | exfalso; exact (Hcur eq_refl) ].
    cbn. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* The [Let]s of [proj_lets] run, each binding its variable to the new
   slot it reads. *)
Lemma proj_lets_run (cs : function_slot) (y : variable)
    (pairs : list (variable * variable * flambda_kind)) :
  ∀ (body : expr) (vars : env) (conts : cont_env) (S : static_env)
    (hp : heap) (rd : read_record),
    NoDup (map (fun '(x, _, _) => x) pairs) ->
    y ∉ map (fun '(x, _, _) => x) pairs ->
    (∀ (x s : variable) (k : flambda_kind),
       (x, s, k) ∈ pairs ->
       ∃ w : value,
         eval_named S vars (Project_value_slot_from cs s (Var y)) = Some w
         ∧ kind_of_value w = k) ->
    ∃ vars1 : env,
      run (length pairs) (State (proj_lets cs (Var y) pairs body) vars conts S
                            hp rd)
      = Step (State body vars1 conts S hp rd)
      ∧ (∀ z : variable,
           z ∉ map (fun '(x, _, _) => x) pairs ->
           lookup z vars1 = lookup z vars)
      ∧ (∀ (x s : variable) (k : flambda_kind) (w : value),
           (x, s, k) ∈ pairs ->
           eval_named S vars (Project_value_slot_from cs s (Var y)) = Some w ->
           lookup x vars1 = Some w).
Proof.
  induction pairs as [ | [[x s] k] pairs IH]; intros body vars conts S hp rd
    Hnd Hy Heval.
  - exists vars. split; [ reflexivity | ]. split; [ reflexivity | ].
    intros x s k w [].
  - cbn [map] in Hnd, Hy.
    inversion Hnd as [ | x0 l0 Hx Hnd' [Heq0 Heq1]]; subst.
    destruct (IH (Let x k (Project_value_slot_from cs s (Var y)) body) vars
                conts S hp rd Hnd')
      as (vars1 & Hrun & Hother & Hbound).
    { intros Hin. apply Hy. right. exact Hin. }
    { intros x1 s1 k1 Hin. exact (Heval x1 s1 k1 (or_intror Hin)). }
    destruct (Heval x s k (or_introl eq_refl)) as (w & Hw & Hk).
    assert (Hw1 : eval_named S vars1 (Project_value_slot_from cs s (Var y))
                  = Some w).
    { rewrite <- Hw. cbn [eval_named eval_simple]. rewrite Hother;
        [ reflexivity | ].
      intros Hin. apply Hy. right. exact Hin. }
    exists ((x, w) :: vars1). split; [ | split ].
    + change (proj_lets cs (Var y) ((x, s, k) :: pairs) body)
        with (proj_lets cs (Var y) pairs
                (Let x k (Project_value_slot_from cs s (Var y)) body)).
      cbn [length]. rewrite <- Nat.add_1_r, run_add, Hrun, run_succ,
        step_let_exec.
      cbn [exec_named]. rewrite Hw1. cbn. rewrite Hk, kind_eqb_refl.
      reflexivity.
    + intros z Hz. rewrite lookup_other
        by (intros ->; apply Hz; left; reflexivity).
      apply Hother. intros Hin. apply Hz. right. exact Hin.
    + intros x' s' k' w' [Heq | Hin] Hw'.
      * injection Heq as -> -> ->. rewrite Hw in Hw'. injection Hw' as <-.
        apply lookup_head.
      * rewrite lookup_other; [ exact (Hbound x' s' k' w' Hin Hw') | ].
        intros ->. apply Hx. apply in_map_iff. exists (x, s', k').
        split; [ reflexivity | exact Hin ].
Qed.

Lemma proj_lets_free (cs : function_slot) (y : simple)
    (pairs : list (variable * variable * flambda_kind)) :
  ∀ body : expr,
    free_continuations_of_expr (proj_lets cs y pairs body)
    = free_continuations_of_expr body
    ∧ binds_statics (proj_lets cs y pairs body) = binds_statics body.
Proof.
  induction pairs as [ | [[x s] k] pairs IH]; intros body;
    [ split; reflexivity | ].
  change (proj_lets cs y ((x, s, k) :: pairs) body)
    with (proj_lets cs y pairs (Let x k (Project_value_slot_from cs s y) body)).
  destruct (IH (Let x k (Project_value_slot_from cs s y) body)) as [H1 H2].
  rewrite H1, H2. split; reflexivity.
Qed.

(* The closure that a call of a value calls: that of a partial
   application, or the value. *)
Definition call_closure (v : value) : value :=
  match partial_application_of v with
  | Some (cl, _, _) => cl
  | None => v
  end.

Lemma partial_application_of_some (v cl : value) (applied : list value)
    (aa : arity) :
  partial_application_of v = Some (cl, applied, aa) ->
  v = Value_partial_application cl applied aa.
Proof. destruct v; cbn; congruence. Qed.

(* ======================================================================== *)
(* Related environments                                                     *)
(* ======================================================================== *)

Section Simulation.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.
Variable CH : variable -> option closure_repr.
Hypothesis CH_M : ∀ (x : variable) (r : closure_repr),
  CH x = Some r -> fields_of M x = Some [].
Hypothesis CH_fresh : ∀ (x : variable) (r : closure_repr),
  CH x = Some r -> among x (field_variables M) = false.
Variable CS : code_id -> option code_info.
Variable RCH : bool -> variable -> code_id -> Prop.
Variable W : variable -> Prop.
Variable REM : code_id -> Prop.

Lemma plain_not_changed (x : variable) :
  plain_variable M x = true -> CH x = None.
Proof.
  intros Hx. destruct (CH x) as [r | ] eqn:E; [ | reflexivity ].
  exfalso. exact (plain_not_unboxed M x [] Hx (CH_M x r E)).
Qed.

Lemma fields_hold_nil (vars' : env) (w : value) : fields_hold vars' [] w.
Proof.
  intros p v k Hp. apply upath_unboxed_inv in Hp as (f & u & p0 & _ & [] & _).
Qed.

(* A variable not among the variables of the fields of an unboxed name
   leaves what they hold as it is. *)
Lemma fields_hold_other (vars' : env) (x : variable) (w' : value)
    (fields : unboxed_fields) (w : value) :
  (∀ (p : list field) (v : variable) (k : flambda_kind),
     upath Value (Unboxed fields) p v k -> v <> x) ->
  fields_hold vars' fields w -> fields_hold ((x, w') :: vars') fields w.
Proof.
  intros Hne Hhold p v k Hp. destruct (Hhold p v k Hp) as (w0 & H1 & H2 & H3).
  exists w0. split; [ exact H1 | split; [ | exact H3 ] ].
  rewrite lookup_other by exact (Hne p v k Hp). exact H2.
Qed.

Lemma env_rel_c_bind (vars vars' : env) (x : variable) (w : value) :
  env_rel_c M CH vars vars' -> plain_variable M x = true ->
  env_rel_c M CH ((x, w) :: vars) ((x, w) :: vars').
Proof.
  intros [Hrel Hch] Hx. split; [ exact (env_rel_bind M _ _ x w Hrel
                                          Hx) | ].
  intros y r w0 Hy Hl.
  assert (Hne : y <> x)
    by (intros ->; rewrite (plain_not_changed x Hx) in Hy; discriminate Hy).
  rewrite lookup_other in Hl |- * by exact Hne. exact (Hch y r w0 Hy Hl).
Qed.

Lemma env_rel_c_bind_changed (vars vars' : env) (x : variable)
    (r : closure_repr) (w w' : value) :
  env_rel_c M CH vars vars' -> CH x = Some r -> crel_exact r w w' ->
  env_rel_c M CH ((x, w) :: vars) ((x, w') :: vars').
Proof.
  intros [[Hplain Hfs] Hch] Hx Hc. split; [ split | ].
  - intros z Hz.
    assert (Hne : z <> x)
      by (intros ->; rewrite (plain_not_changed x Hz) in Hx; discriminate Hx).
    rewrite !lookup_other by exact Hne. exact (Hplain z Hz).
  - intros y fy w0 Hy Hl. destruct (Pos.eq_dec y x) as [-> | Hne].
    + rewrite (CH_M x r Hx) in Hy. injection Hy as <-. apply fields_hold_nil.
    + rewrite lookup_other in Hl by exact Hne.
      apply fields_hold_other; [ | exact (Hfs y fy w0 Hy Hl) ].
      intros p v k Hp ->.
      pose proof (field_variables_in M y fy x Hy
                    (upath_leaf_in fy p x k Hp)) as Hin.
      apply among_In in Hin. rewrite (CH_fresh x r Hx) in Hin.
      discriminate Hin.
  - intros y r' w0 Hy Hl. destruct (Pos.eq_dec y x) as [-> | Hne].
    + rewrite lookup_head in Hl |- *. injection Hl as <-.
      rewrite Hx in Hy. injection Hy as <-. exists w'.
      split; [ reflexivity | exact Hc ].
    + rewrite lookup_other in Hl |- * by exact Hne. exact (Hch y r' w0 Hy Hl).
Qed.

(* The run of the second binds the variables of the fields of [x], and
   nothing else; the first binds [x], an unboxed name. *)
Lemma env_rel_c_bind_fields (vars vars' vars1 : env) (x : variable)
    (fields : unboxed_fields) (w : value) :
  env_rel_c M CH vars vars' -> fields_of M x = Some fields -> CH x = None ->
  (∀ z : variable,
     z ∉ leaf_variables fields -> lookup z vars1 = lookup z vars') ->
  fields_hold vars1 fields w ->
  env_rel_c M CH ((x, w) :: vars) vars1.
Proof.
  intros [Hrel Hch] Hx Hxc Hother Hhold. split.
  - exact (env_rel_bind_fields M M_distinct vars vars' vars1 x fields w Hrel
             Hx Hother Hhold).
  - intros y r w0 Hy Hl.
    assert (Hne : y <> x) by (intros ->; rewrite Hxc in Hy; discriminate Hy).
    rewrite lookup_other in Hl by exact Hne.
    rewrite Hother; [ exact (Hch y r w0 Hy Hl) | ].
    intros Hin. pose proof (field_variables_in M x fields y Hx Hin) as Hfv.
    apply among_In in Hfv. rewrite (CH_fresh y r Hy) in Hfv.
    discriminate Hfv.
Qed.

Lemma env_rel_c_empty : env_rel_c M CH [] [].
Proof.
  split; [ exact (env_rel_empty M) | ]. intros x r w _ Hl. discriminate Hl.
Qed.



(* The new slots of a set related are bound in the second run to what
   the old slots hold in the first, at the paths of their fields. *)
Lemma new_slots_eval (vars vars' : env) (r : closure_repr)
    (vs : list (value_slot * simple)) (vs1 : list (value_slot * value)) :
  env_rel M vars vars' -> Forall (set_field_ok M CH vs) (cr_fields r) ->
  NoDup (map fst (new_slots M r vs)) ->
  eval_value_slots vars vs = Some vs1 ->
  ∃ vs2 : list (value_slot * value),
    eval_value_slots vars' (new_slots M r vs) = Some vs2
    ∧ ∀ (w : value_slot) (u : unboxed_field),
        (Field_value_slot w, u) ∈ cr_fields r ->
        ∃ v0 : value, lookup w vs1 = Some v0 ∧ slot_holds vs2 u v0.
Proof.
  intros Hrel Hok Hnd Hvs1. rewrite Forall_forall in Hok.
  (* What each field makes. *)
  assert (Hfield : ∀ (w : value_slot) (u : unboxed_field),
            (Field_value_slot w, u) ∈ cr_fields r ->
            ∃ (a : simple) (v0 : value),
              lookup w vs = Some a ∧ eval_simple vars a = Some v0
              ∧ lookup w vs1 = Some v0
              ∧ match u with
                | Not_unboxed s =>
                  new_slot M vs (Field_value_slot w, u) = [(s, a)]
                  ∧ eval_simple vars' a = Some v0
                | Unboxed nested =>
                  ∃ (b : variable) (fb : unboxed_fields)
                    (pairs : list (variable * variable * flambda_kind)),
                    new_slot M vs (Field_value_slot w, u)
                    = map (fun '(s, x, _) => (s, Var x)) pairs
                    ∧ unboxed_field_pairs Value (Unboxed nested) (Unboxed fb)
                      = Some pairs
                    ∧ fields_hold vars' fb v0
                end).
  { intros w u Hin. specialize (Hok _ Hin). cbn [set_field_ok] in Hok.
    destruct Hok as (a & Ha & Hu).
    destruct (eval_value_slots_lookup vars vs vs1 w a Hvs1 Ha)
      as (v0 & Hv0 & Hl0).
    exists a, v0. split; [ exact Ha | split; [ exact Hv0 | ] ].
    split; [ exact Hl0 | ]. unfold new_slot. rewrite Ha.
    destruct u as [s | nested].
    - split; [ reflexivity | ].
      rewrite <- (plain_simple_eval M vars vars' a Hrel Hu). exact Hv0.
    - destruct Hu as (b & fb & pairs & -> & _ & Hb & Hp).
      exists b, fb, pairs. rewrite Hb, Hp.
      split; [ reflexivity | split; [ reflexivity | ] ].
      exact (proj2 Hrel b fb v0 Hb Hv0). }
  destruct (eval_value_slots_some vars' (new_slots M r vs)) as (vs2 & Hvs2).
  { intros s a' Hin. unfold new_slots in Hin.
    apply in_flat_map in Hin as ([f u] & Hfu & Hin).
    destruct f as [ | w | | | | | | | ];
      try (cbn [new_slot] in Hin; destruct Hin; fail).
    destruct (Hfield w u Hfu) as (a & v0 & _ & _ & _ & Hcase).
    destruct u as [s0 | nested].
    - destruct Hcase as [Heq Hv']. rewrite Heq in Hin.
      destruct Hin as [Heq' | []]. injection Heq' as <- <-.
      exists v0. exact Hv'.
    - destruct Hcase as (b & fb & pairs & Heq & Hp & Hhold).
      rewrite Heq in Hin. apply in_map_iff in Hin as ([[s1 x] k] & Heq' & Hin).
      injection Heq' as <- <-.
      destruct (pairs_hold vars' nested fb v0 pairs Hp Hhold)
        as (_ & Hseconds & _).
      destruct (Hseconds s1 x k Hin) as (_ & w' & Hl' & _).
      exists w'. exact Hl'. }
  exists vs2. split; [ exact Hvs2 | ].
  intros w u Hin.
  destruct (Hfield w u Hin) as (a & v0 & _ & _ & Hl0 & Hcase).
  exists v0. split; [ exact Hl0 | ].
  assert (Hsub : ∀ e, e ∈ new_slot M vs (Field_value_slot w, u) ->
                   e ∈ new_slots M r vs)
    by (intros e He; unfold new_slots; apply in_flat_map;
        exists (Field_value_slot w, u); split; assumption).
  destruct u as [s | nested]; cbn [slot_holds].
  - destruct Hcase as [Heq Hv'].
    apply (eval_value_slots_in vars' _ vs2 s a v0 Hvs2 Hnd); [ | exact Hv' ].
    apply Hsub. rewrite Heq. left. reflexivity.
  - destruct Hcase as (b & fb & pairs & Heq & Hp & Hhold).
    destruct (pairs_hold vars' nested fb v0 pairs Hp Hhold)
      as (_ & Hseconds & Hpaths).
    intros p s k Hpath.
    destruct (Hpaths p s k Hpath) as (x & w' & Hin' & Hl' & Hpw).
    destruct (Hseconds s x k Hin') as (_ & w'' & Hl'' & Hk).
    rewrite Hl' in Hl''. injection Hl'' as <-.
    exists w'. split; [ exact Hpw | split; [ | exact Hk ] ].
    apply (eval_value_slots_in vars' _ vs2 s (Var x) w' Hvs2 Hnd);
      [ | exact Hl' ].
    apply Hsub. rewrite Heq. apply in_map_iff. exists (s, x, k).
    split; [ reflexivity | exact Hin' ].
Qed.


(* The keys of the value slots of a closure made: those of its slots. *)
Lemma eval_value_slots_keys (vars : env) (l : list (value_slot * simple))
    (vs : list (value_slot * value)) :
  eval_value_slots vars l = Some vs -> map fst vs = map fst l.
Proof.
  revert vs. induction l as [ | [w0 s0] l IH]; intros vs Hl;
    [ injection Hl as <-; reflexivity | ].
  cbn [eval_value_slots] in Hl.
  destruct (eval_simple vars s0) as [v0 | ]; [ | discriminate Hl ].
  destruct (eval_value_slots vars l) as [vs0 | ] eqn:Hvs0;
    [ | discriminate Hl ].
  cbn in Hl. injection Hl as <-. cbn [map fst]. f_equal. exact (IH vs0 eq_refl).
Qed.

(* The new slots of the fields of a set related are the leaves of the
   fields, in their order. *)
Lemma new_slots_keys_c (vs : list (value_slot * simple))
    (fields : unboxed_fields) :
  Forall (set_field_ok M CH vs) fields ->
  map fst (flat_map (new_slot M vs) fields) = map fst (fields_leaves fields).
Proof.
  induction 1 as [ | [f u] fields Hfu _ IH]; [ reflexivity | ].
  cbn [flat_map]. rewrite leaves_cons, !map_app, IH. f_equal.
  destruct f; try contradiction. cbn [set_field_ok] in Hfu.
  destruct Hfu as (a & Ha & Hu). unfold new_slot. rewrite Ha.
  destruct u as [s | nested]; [ reflexivity | ].
  destruct Hu as (b & fb & pairs & -> & _ & Hb & Hp). rewrite Hb, Hp.
  rewrite map_map. cbn [field_kind].
  destruct (pairs_spec _ _ _ _ Hp) as (_ & _ & Hc & _).
  rewrite <- Hc. apply map_ext. intros [[s x] k]. reflexivity.
Qed.

(* The closures of a set whose representation changes, made by the two
   runs from related environments, are related by [crel_exact]. *)
Lemma set_crel_exact (vars : env) (r : closure_repr)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (vs1 vs2 : list (value_slot * value))
    (code : code_id) :
  lookup (cr_current r) decls = Some code ->
  lookup (current_slot r) (rename_function_decls r decls)
  = lookup (cr_current r) decls ->
  Forall (set_field_ok M CH vs) (cr_fields r) ->
  NoDup (map fst (new_slots M r vs)) ->
  eval_value_slots vars (new_slots M r vs) = Some vs2 ->
  (∀ (w : value_slot) (u : unboxed_field),
     (Field_value_slot w, u) ∈ cr_fields r ->
     ∃ v0 : value, lookup w vs1 = Some v0 ∧ slot_holds vs2 u v0) ->
  crel_exact r (Value_closure_of_set (cr_current r) decls vs1)
    (Value_closure_of_set (current_slot r) (rename_function_decls r decls)
       vs2).
Proof.
  intros Hcode Hdecls Hok Hnd Hvs2 Hslots.
  assert (Hkeys : map fst vs2 = map fst (fields_leaves (cr_fields r))).
  { rewrite (eval_value_slots_keys _ _ _ Hvs2). unfold new_slots.
    exact (new_slots_keys_c vs (cr_fields r) Hok). }
  split.
  - exists decls, vs1, vs2.
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ exact Hdecls | exact Hslots ].
  - exists decls, vs1, vs2.
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ rewrite Hcode; discriminate | split; [ exact Hkeys | ] ].
    split; [ rewrite (eval_value_slots_keys _ _ _ Hvs2); exact Hnd | ].
    intros [f u] Hin. rewrite Forall_forall in Hok.
    pose proof (Hok _ Hin) as Hf. destruct f; try contradiction.
    eexists _, _. reflexivity.
Qed.

(* The [Let]s that the unboxing makes of the fields of [x], of simples,
   run, and bind them as the relation asks. *)
Lemma lets_case_c (vars vars' : env) (x : variable) (fields : unboxed_fields)
    (w : value) (l : list (variable * flambda_kind * simple)) (body : expr)
    (conts : cont_env) (S : static_env) (hp : heap) (rcd : read_record) :
  env_rel_c M CH vars vars' -> fields_of M x = Some fields -> CH x = None ->
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
    run (length l) (State (lets l body) vars' conts S hp rcd)
    = Step (State body vars1 conts S hp rcd)
    ∧ env_rel_c M CH ((x, w) :: vars) vars1.
Proof.
  intros Hvars Hx Hxc Hvl Hread Heval Hpaths.
  destruct (lets_run l body vars' conts S hp rcd)
    as (vars1 & Hrun & Hother & Hbound).
  - rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hx).
  - intros v kind s v' Hin Hv'. rewrite Hvl in Hv'.
    exact (Hread v kind s v' Hin Hv').
  - exact Heval.
  - exists vars1. split; [ exact Hrun | ].
    apply (env_rel_c_bind_fields vars vars' vars1 x fields w Hvars Hx Hxc).
    + intros z Hz. apply Hother. rewrite Hvl. exact Hz.
    + intros p v k Hp. destruct (Hpaths p v k Hp) as (s & w' & Hin & Hs & Hpw).
      destruct (Heval v k s Hin) as (w'' & Hs' & Hk).
      rewrite Hs in Hs'. injection Hs' as <-.
      exists w'. split; [ exact Hpw | ].
      split; [ exact (Hbound v k s w' Hin Hs) | exact Hk ].
Qed.

(* The [Let]s of [proj_lets] of the fields of [x] run, and bind them as
   the relation asks. *)
Lemma proj_lets_case (vars vars' : env) (x y : variable)
    (fields : unboxed_fields) (w : value) (cs : function_slot)
    (pairs : list (variable * variable * flambda_kind)) (body : expr)
    (conts : cont_env) (S : static_env) (hp : heap) (rcd : read_record) :
  env_rel_c M CH vars vars' -> fields_of M x = Some fields -> CH x = None ->
  map (fun '(a, _, _) => a) pairs = leaf_variables fields ->
  y ∉ leaf_variables fields ->
  (∀ (v s : variable) (k : flambda_kind),
     (v, s, k) ∈ pairs ->
     ∃ w' : value,
       eval_named S vars' (Project_value_slot_from cs s (Var y)) = Some w'
       ∧ kind_of_value w' = k) ->
  (∀ (p : list field) (v : variable) (k : flambda_kind),
     upath Value (Unboxed fields) p v k ->
     ∃ (s : variable) (w' : value),
       (v, s, k) ∈ pairs
       ∧ eval_named S vars' (Project_value_slot_from cs s (Var y)) = Some w'
       ∧ path_value p w = Some w') ->
  ∃ vars1 : env,
    run (length pairs) (State (proj_lets cs (Var y) pairs body) vars' conts S
                          hp rcd)
    = Step (State body vars1 conts S hp rcd)
    ∧ env_rel_c M CH ((x, w) :: vars) vars1.
Proof.
  intros Hvars Hx Hxc Hvl Hy Heval Hpaths.
  destruct (proj_lets_run cs y pairs body vars' conts S hp rcd)
    as (vars1 & Hrun & Hother & Hbound).
  - rewrite Hvl. exact (leaves_nodup M M_distinct x fields Hx).
  - rewrite Hvl. exact Hy.
  - exact Heval.
  - exists vars1. split; [ exact Hrun | ].
    apply (env_rel_c_bind_fields vars vars' vars1 x fields w Hvars Hx Hxc).
    + intros z Hz. apply Hother. rewrite Hvl. exact Hz.
    + intros p v k Hp. destruct (Hpaths p v k Hp) as (s & w' & Hin & Hs & Hpw).
      destruct (Heval v s k Hin) as (w'' & Hs' & Hk).
      rewrite Hs in Hs'. injection Hs' as <-.
      exists w'. split; [ exact Hpw | ].
      split; [ exact (Hbound v s k w' Hin Hs) | exact Hk ].
Qed.

(* ======================================================================== *)
(* Shapes of related terms                                                  *)
(* ======================================================================== *)

Lemma code_free_conts (c : code) :
  free_continuations_of_code c
  = without [code_return_continuation c; code_exn_continuation c]
      (free_continuations_of_expr (code_body c)).
Proof. destruct c. reflexivity. Qed.

(* A free continuation of the second term is one of the first or dead;
   the second binds statics only where the first does. *)
Definition fc_below (D : list continuation) (l l' : list continuation)
    : Prop :=
  ∀ k : continuation, k ∈ l' -> k ∈ l ∨ k ∈ D.

Lemma orb_impl (a b a' b' : bool) :
  (a' = true -> a = true) -> (b' = true -> b = true) ->
  a' || b' = true -> a || b = true.
Proof.
  intros Ha Hb H. apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (Ha H) | right; exact (Hb H) ].
Qed.

Lemma rr_shape :
  (∀ (D : list continuation) (p q : expr),
     rr M CH CS RCH W REM D p q ->
     fc_below D (free_continuations_of_expr p) (free_continuations_of_expr q)
     ∧ (binds_statics q = true -> binds_statics p = true))
  ∧ (∀ (D : list continuation)
       (hs hs' : list (continuation * continuation_handler)),
       rr_handlers M CH CS RCH W REM D hs hs' ->
       map fst hs = map fst hs'
       ∧ fc_below D
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs)
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs')
       ∧ (handlers_bind_statics hs' = true ->
          handlers_bind_statics hs = true))
  ∧ (∀ bs bs' : list static_binding,
       rr_bindings M CH CS RCH W REM bs bs' ->
       incl (flat_map free_continuations_of_binding bs')
         (flat_map free_continuations_of_binding bs)
       ∧ (existsb binding_binds bs' = true ->
          existsb binding_binds bs = true)).
Proof.
  apply (rr_comb M CH CS RCH W REM
    (fun D p q _ =>
       fc_below D (free_continuations_of_expr p) (free_continuations_of_expr q)
       ∧ (binds_statics q = true -> binds_statics p = true))
    (fun D hs hs' _ =>
       map fst hs = map fst hs'
       ∧ fc_below D
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs)
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs')
       ∧ (handlers_bind_statics hs' = true ->
          handlers_bind_statics hs = true))
    (fun bs bs' _ =>
       incl (flat_map free_continuations_of_binding bs')
         (flat_map free_continuations_of_binding bs)
       ∧ (existsb binding_binds bs' = true ->
          existsb binding_binds bs = true)));
    intros; repeat match goal with
                   | H : _ ∧ _ |- _ => destruct H
                   end;
    unfold fc_below in *.
  (* The [Let]s of both, and the reads of new slots. *)
  1, 2, 6, 7, 8: cbn [free_continuations_of_expr binds_statics];
    split; assumption.
  (* The [Let]s that the unboxing makes. *)
  1: destruct (lets_free_continuations (concat ls) body') as [H1 H2].
  2, 3: destruct (lets_free_continuations (pairs_lets pairs) body')
          as [H1 H2].
  4: destruct (proj_lets_free (current_slot r) (Var y) pairs body')
       as [H1 H2].
  1, 2, 3, 4: cbn [free_continuations_of_expr binds_statics];
    rewrite H1, H2; split; assumption.
  (* [Let_static]. *)
  - cbn [free_continuations_of_expr binds_statics].
    split; [ | apply orb_impl; assumption ].
    intros k0 Hk0. apply in_app_or in Hk0 as [Hk0 | Hk0].
    + left. apply in_or_app. left. auto.
    + match goal with
      | H : ∀ k, k ∈ free_continuations_of_expr body' -> _ |- _ =>
        destruct (H k0 Hk0) as [Hk1 | Hk1]
      end; [ left; apply in_or_app; right; exact Hk1 | right; exact Hk1 ].
  (* [Let_cont]. *)
  - cbn [free_continuations_of_expr binds_statics free_continuations_of_handler
         handler_binds_statics].
    split; [ | apply orb_impl; assumption ].
    intros k0 Hk0. apply in_app_or in Hk0 as [Hk0 | Hk0].
    + match goal with
      | H : ∀ k, k ∈ free_continuations_of_expr h' -> _ |- _ =>
        destruct (H k0 Hk0) as [Hk1 | Hk1]
      end; [ left; apply in_or_app; left; exact Hk1 | right; exact Hk1 ].
    + apply in_without_iff in Hk0 as [Hk0 Hnk].
      match goal with
      | H : ∀ k, k ∈ free_continuations_of_expr body' -> _ |- _ =>
        destruct (H k0 Hk0) as [Hk1 | Hk1]
      end.
      * left. apply in_or_app. right. apply in_without_iff.
        split; assumption.
      * right. apply in_without_iff in Hk1 as [Hk1 _]. exact Hk1.
  - cbn [free_continuations_of_expr binds_statics].
    split.
    + intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hnk].
      match goal with
      | H : map fst hs = map fst hs' |- _ => rewrite <- H in Hnk
      end.
      apply in_app_or in Hk0 as [Hk0 | Hk0];
        match goal with
        | H : ∀ k, k ∈ ?l -> _ , Hk : k0 ∈ ?l |- _ =>
          destruct (H k0 Hk) as [Hk1 | Hk1]
        end;
        try (right; apply in_without_iff in Hk1 as [Hk1 _]; exact Hk1);
        left; apply in_without_iff; (split; [ | exact Hnk ]);
        apply in_or_app; [ left | right ]; exact Hk1.
    + apply orb_impl; [ | assumption ].
      unfold handlers_bind_statics in *. assumption.
  (* The others are the same in both. *)
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  (* The calls of code that never returns. *)
  - cbn [free_continuations_of_expr binds_statics ap_continuation
         ap_exn_continuation with_continuation].
    split; [ | intros Hb; discriminate Hb ].
    intros k0 [<- | [<- | []]]; [ right; assumption | ].
    left. right. left. reflexivity.
  - cbn [free_continuations_of_expr binds_statics ap_continuation
         ap_exn_continuation with_continuation free_continuations_of_handler
         handler_binds_statics app].
    split; [ | intros Hb; discriminate Hb ].
    intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hnk].
    destruct Hk0 as [<- | [<- | []]];
      [ exfalso; apply Hnk; left; reflexivity | ].
    left. right. left. reflexivity.
  - cbn [free_continuations_of_expr binds_statics ap_continuation
         ap_exn_continuation with_continuation free_continuations_of_handler
         handler_binds_statics app].
    split; [ | intros Hb; discriminate Hb ].
    intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hnk].
    destruct Hk0 as [<- | [<- | []]];
      [ exfalso; apply Hnk; left; reflexivity | ].
    left. apply in_or_app. right. apply in_without_iff.
    split; [ right; left; reflexivity | ].
    intros [Heq | []]. auto.
  (* A handler that only the first run binds. *)
  - cbn [free_continuations_of_expr binds_statics].
    split; [ | intros Hb; apply orb_true_iff; right; auto ].
    intros k0 Hk0.
    match goal with
    | H : ∀ k1, k1 ∈ free_continuations_of_expr body' -> _ |- _ =>
      destruct (H k0 Hk0) as [Hk1 | [Hk1 | Hk1]]
    end.
    + left. apply in_or_app. right. apply in_without_iff.
      split; [ exact Hk1 | ]. intros [Heq | []]. subst k0. contradiction.
    + subst k0. contradiction.
    + right. exact Hk1.
  (* A call through a wrapper of each run. *)
  - cbn [free_continuations_of_expr binds_statics ap_continuation
         ap_exn_continuation with_continuation free_continuations_of_handler
         handler_binds_statics wrapper_body ac_continuation app].
    split; [ | intros Hb; discriminate Hb ].
    intros k0 [<- | Hk0]; [ left; left; reflexivity | ].
    apply in_without_iff in Hk0 as [Hk0 Hnk].
    destruct Hk0 as [<- | [<- | []]];
      [ exfalso; apply Hnk; left; reflexivity | ].
    left. right. apply in_without_iff.
    split; [ right; left; reflexivity | ].
    intros [Heq | []]. auto.
  (* A call that may over-apply. *)
  - split; [ intros k0 Hk0; left; exact Hk0 | intros Hb; exact Hb ].
  (* Handlers. *)
  - split; [ reflexivity | ].
    split; [ intros k0 [] | intros Hb; exact Hb ].
  - cbn [map fst flat_map handlers_bind_statics existsb].
    split; [ f_equal; assumption | ].
    split.
    + intros k0 Hk0. cbn [free_continuations_of_handler] in Hk0 |- *.
      apply in_app_or in Hk0 as [Hk0 | Hk0];
        match goal with
        | H : ∀ k, k ∈ ?l -> _ , Hk : k0 ∈ ?l |- _ =>
          destruct (H k0 Hk) as [Hk1 | Hk1]
        end;
        try (right; exact Hk1); left; apply in_or_app;
        [ left | right ]; exact Hk1.
    + unfold handlers_bind_statics in *. cbn [existsb].
      apply orb_impl; assumption.
  (* Bindings. *)
  - split; [ intros k0 []| intros Hb; exact Hb ].
  - cbn [flat_map existsb].
    split; [ apply incl_app_app; [ apply incl_refl | assumption ] | ].
    apply orb_impl; [ intros Hb; exact Hb | assumption ].
  - cbn [flat_map existsb free_continuations_of_binding binding_binds].
    split; assumption.
  - change (flat_map free_continuations_of_binding (Bind_code id c :: bs))
      with (free_continuations_of_code c
            ++ flat_map free_continuations_of_binding bs).
    change (flat_map free_continuations_of_binding (Bind_code id c' :: bs'))
      with (free_continuations_of_code c'
            ++ flat_map free_continuations_of_binding bs').
    cbn [existsb binding_binds orb].
    split; [ | intros Hb; reflexivity ].
    rewrite !code_free_conts.
    match goal with
    | H : code_frame _ _ _ _ _ _ _ |- _ =>
      destruct H as (_ & _ & _ & Hret & Hexn & _)
    end.
    rewrite Hret, Hexn. apply incl_app_app; [ | assumption ].
    intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hnk].
    match goal with
    | H : ∀ k, k ∈ free_continuations_of_expr (code_body c') -> _ |- _ =>
      destruct (H k0 Hk0) as [Hk1 | Hk1]
    end.
    + apply in_without_iff. split; assumption.
    + exfalso. apply Hnk. unfold code_dead in Hk1.
      destruct (ci_never ci); [ | destruct Hk1 ].
      destruct Hk1 as [<- | []]. left. reflexivity.
Qed.

(* The dead continuations count only where free in the second term. *)
Definition dead_agree (D D' : list continuation) (l : list continuation)
    : Prop :=
  ∀ k : continuation, k ∈ l -> k ∈ D <-> k ∈ D'.

Lemma dead_agree_incl (D D' l l' : list continuation) :
  incl l l' -> dead_agree D D' l' -> dead_agree D D' l.
Proof. intros Hi H k Hk. exact (H k (Hi k Hk)). Qed.

Lemma dead_agree_without (D D' bound l : list continuation) :
  dead_agree D D' (without bound l) ->
  dead_agree (without bound D) (without bound D') l.
Proof.
  intros H k Hk. rewrite !in_without_iff.
  split; intros [Hd Hb]; (split; [ | exact Hb ]);
    apply (H k (proj2 (in_without_iff _ _ _) (conj Hk Hb))); exact Hd.
Qed.

Lemma dead_agree_cons (D D' l : list continuation) (k : continuation) :
  dead_agree D D' l -> dead_agree (k :: D) (k :: D') l.
Proof.
  intros H k0 Hk0. split; intros [He | Hin];
    [ left; exact He | right; exact (proj1 (H k0 Hk0) Hin)
    | left; exact He | right; exact (proj2 (H k0 Hk0) Hin) ].
Qed.

Lemma rr_dead_ext :
  (∀ (D : list continuation) (p q : expr),
     rr M CH CS RCH W REM D p q ->
     ∀ D' : list continuation,
       dead_agree D D' (free_continuations_of_expr q) ->
       rr M CH CS RCH W REM D' p q)
  ∧ (∀ (D : list continuation)
       (hs hs' : list (continuation * continuation_handler)),
       rr_handlers M CH CS RCH W REM D hs hs' ->
       ∀ D' : list continuation,
         dead_agree D D'
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs') ->
         rr_handlers M CH CS RCH W REM D' hs hs')
  ∧ (∀ bs bs' : list static_binding,
       rr_bindings M CH CS RCH W REM bs bs' -> True).
Proof.
  apply (rr_comb M CH CS RCH W REM
    (fun D p q _ =>
       ∀ D' : list continuation,
         dead_agree D D' (free_continuations_of_expr q) ->
         rr M CH CS RCH W REM D' p q)
    (fun D hs hs' _ =>
       ∀ D' : list continuation,
         dead_agree D D'
           (flat_map (fun '(_, h) => free_continuations_of_handler h) hs') ->
         rr_handlers M CH CS RCH W REM D' hs hs')
    (fun _ _ _ => True)); intros; try exact I.
  (* The [Let]s of both, and the reads of new slots. *)
  1, 2, 6, 7, 8:
    econstructor; try eassumption;
    match goal with
    | IH : ∀ D', dead_agree _ D' _ -> _ |- _ => apply IH
    end;
    assumption.
  (* The [Let]s that the unboxing makes. *)
  1: destruct (lets_free_continuations (concat ls) body') as [H1 _].
  2, 3: destruct (lets_free_continuations (pairs_lets pairs) body')
          as [H1 _].
  4: destruct (proj_lets_free (current_slot r) (Var y) pairs body')
       as [H1 _].
  1, 2, 3, 4:
    econstructor; try eassumption;
    match goal with
    | IH : ∀ D', dead_agree _ D' _ -> _ |- _ => apply IH
    end;
    match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr] in H; rewrite H1 in H; exact H
    end.
  (* [Let_static]. *)
  - match goal with
    | H : dead_agree _ _ _, IH : ∀ D', dead_agree _ D' _ -> _ |- _ =>
      apply rr_static; [ assumption | apply IH ];
      eapply dead_agree_incl; [ | exact H ];
      intros k0 Hk0; apply in_or_app; right; exact Hk0
    end.
  (* [Let_cont]. *)
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr free_continuations_of_handler] in H
    end.
    apply rr_let_cont; [ assumption | | ].
    + match goal with
      | IH : ∀ D', dead_agree D D' (free_continuations_of_expr h') -> _
        |- _ => apply IH
      end.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_or_app. left. exact Hk0.
    + match goal with
      | IH : ∀ D', dead_agree (without [k] D) D'
                     (free_continuations_of_expr body') -> _ |- _ => apply IH
      end.
      apply dead_agree_without. eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_or_app. right. exact Hk0.
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr] in H
    end.
    match goal with
    | Hhs : rr_handlers M CH CS RCH W REM _ hs hs' |- _ =>
      pose proof (proj1 (proj1 (proj2 rr_shape) _ _ _ Hhs)) as Hkeys
    end.
    apply rr_let_cont_rec.
    + match goal with
      | IH : ∀ D', dead_agree _ D' (flat_map _ hs') -> _ |- _ => apply IH
      end.
      rewrite Hkeys. apply dead_agree_without.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hn].
      apply in_without_iff. split; [ | exact Hn ].
      apply in_or_app. left. exact Hk0.
    + match goal with
      | IH : ∀ D', dead_agree _ D' (free_continuations_of_expr body') -> _
        |- _ => apply IH
      end.
      rewrite Hkeys. apply dead_agree_without.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_without_iff in Hk0 as [Hk0 Hn].
      apply in_without_iff. split; [ | exact Hn ].
      apply in_or_app. right. exact Hk0.
  (* The others read [D] at continuations free in the second term. *)
  - apply rr_apply_cont; [ assumption | ].
    intros Hin. match goal with
                | H : dead_agree _ _ _ |- _ =>
                  apply (proj2 (H _ (or_introl eq_refl))) in Hin
                end.
    contradiction.
  - apply rr_switch; [ assumption | assumption | ].
    intros arm Harm Hin.
    assert (Hf : ac_continuation (snd arm)
                 ∈ free_continuations_of_expr (Switch scrutinee arms))
      by (cbn [free_continuations_of_expr]; apply in_flat_map;
          exists arm; split; [ exact Harm | left; reflexivity ]).
    match goal with
    | H : dead_agree _ _ _ |- _ => apply (proj2 (H _ Hf)) in Hin
    end.
    match goal with
    | H : ∀ arm, arm ∈ arms -> _ |- _ => exact (H arm Harm Hin)
    end.
  - apply rr_invalid.
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr] in H
    end.
    eapply rr_call; try eassumption.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_introl eq_refl))) in Hin
                  end.
      contradiction.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_intror (or_introl eq_refl)))) in Hin
                  end.
      contradiction.
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr] in H
    end.
    eapply rr_call_ind; try eassumption.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_introl eq_refl))) in Hin
                  end.
      contradiction.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_intror (or_introl eq_refl)))) in Hin
                  end.
      contradiction.
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr with_continuation ap_continuation
           ap_exn_continuation] in H
    end.
    apply rr_call_never; try assumption.
    + match goal with
      | H : dead_agree _ _ _ |- _ =>
        apply (proj1 (H _ (or_introl eq_refl))); assumption
      end.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_intror (or_introl eq_refl)))) in Hin
                  end.
      contradiction.
  - apply rr_call_never_r; try assumption.
    intros Hin.
    match goal with
    | H : dead_agree _ _ (free_continuations_of_expr ?q) |- _ =>
      assert (Hf : ap_exn_continuation ap ∈ free_continuations_of_expr q)
        by (cbn [free_continuations_of_expr free_continuations_of_handler app
                 with_continuation ap_continuation ap_exn_continuation];
            apply in_without_iff; split; [ right; left; reflexivity | ];
            intros [He | []]; auto);
      apply (proj2 (H _ Hf)) in Hin
    end.
    contradiction.
  - apply rr_call_never_wraps; try assumption.
    intros Hin.
    match goal with
    | H : dead_agree _ _ (free_continuations_of_expr ?q) |- _ =>
      assert (Hf : ap_exn_continuation ap ∈ free_continuations_of_expr q)
        by (cbn [free_continuations_of_expr free_continuations_of_handler app
                 with_continuation ap_continuation ap_exn_continuation];
            apply in_without_iff; split; [ right; left; reflexivity | ];
            intros [He | []]; auto);
      apply (proj2 (H _ Hf)) in Hin
    end.
    contradiction.
  - apply rr_let_cont_never; [ assumption | ].
    match goal with
    | IH : ∀ D', dead_agree _ D' _ -> _ |- _ => apply IH
    end.
    apply dead_agree_cons. assumption.
  - match goal with
    | |- rr _ _ _ _ _ _ _
           (Let_cont_non_recursive _ (Handler _ _ (wrapper_body ?k _))
              (Apply (with_continuation ?ap _))) ?q =>
      assert (Hf1 : k ∈ free_continuations_of_expr q)
        by (cbn [free_continuations_of_expr free_continuations_of_handler
                 wrapper_body ac_continuation app]; left; reflexivity);
      assert (Hf2 : ap_exn_continuation ap ∈ free_continuations_of_expr q)
        by (cbn [free_continuations_of_expr free_continuations_of_handler
                 wrapper_body ac_continuation app with_continuation
                 ap_continuation ap_exn_continuation];
            right; apply in_without_iff;
            split; [ right; left; reflexivity | ];
            intros [He | []]; auto)
    end.
    eapply rr_call_wraps; try eassumption.
    + intros Hin.
      match goal with
      | H : dead_agree _ _ _ |- _ => apply (proj2 (H _ Hf1)) in Hin
      end.
      contradiction.
    + intros Hin.
      match goal with
      | H : dead_agree _ _ _ |- _ => apply (proj2 (H _ Hf2)) in Hin
      end.
      contradiction.
  - match goal with
    | H : dead_agree _ _ _ |- _ =>
      cbn [free_continuations_of_expr] in H
    end.
    eapply rr_call_over; try eassumption.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_introl eq_refl))) in Hin
                  end.
      contradiction.
    + intros Hin. match goal with
                  | H : dead_agree _ _ _ |- _ =>
                    apply (proj2 (H _ (or_intror (or_introl eq_refl)))) in Hin
                  end.
      contradiction.
  (* Handlers. *)
  - apply rr_handlers_nil.
  - cbn [flat_map free_continuations_of_handler] in *.
    apply rr_handlers_cons; [ assumption | | ].
    + match goal with
      | IH : ∀ D', dead_agree _ D' (free_continuations_of_expr h') -> _
        |- _ => apply IH
      end.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_or_app. left. exact Hk0.
    + match goal with
      | IH : ∀ D', dead_agree _ D' (flat_map _ hs') -> _ |- _ => apply IH
      end.
      eapply dead_agree_incl; [ | eassumption ].
      intros k0 Hk0. apply in_or_app. right. exact Hk0.
Qed.

(* A continuation not free in the second term may be dead. *)
Lemma rr_dead (D : list continuation) (p q : expr) (k : continuation) :
  rr M CH CS RCH W REM D p q -> k ∉ free_continuations_of_expr q ->
  rr M CH CS RCH W REM (k :: D) p q.
Proof.
  intros H Hk. apply (proj1 rr_dead_ext D p q H). intros k0 Hk0.
  split; [ intros Hin; right; exact Hin | ].
  intros [<- | Hin]; [ contradiction | exact Hin ].
Qed.

(* The same, [k] shadowed before. *)
Lemma rr_dead_shadow (D : list continuation) (p q : expr)
    (k : continuation) :
  rr M CH CS RCH W REM (without [k] D) p q ->
  k ∉ free_continuations_of_expr q ->
  rr M CH CS RCH W REM (k :: D) p q ∧ rr M CH CS RCH W REM D p q.
Proof.
  intros H Hk. split; apply (proj1 rr_dead_ext _ p q H); intros k0 Hk0;
    rewrite in_without_iff;
    (assert (Hne : k0 <> k) by (intros ->; contradiction)).
  - split; [ intros [Hin _]; right; exact Hin | ].
    intros [<- | Hin]; [ contradiction | ].
    split; [ exact Hin | intros [He | []]; exact (Hne (eq_sym He)) ].
  - split; [ intros [Hin _]; exact Hin | ].
    intros Hin.
    split; [ exact Hin | intros [He | []]; exact (Hne (eq_sym He)) ].
Qed.

(* ======================================================================== *)
(* Statics and continuations                                                *)
(* ======================================================================== *)

Lemma closure_of_values (S S' : static_env) (v : value) :
  static_values S' = static_values S -> closure_of S' v = closure_of S v.
Proof.
  intros Hv. destruct v; simpl; try reflexivity. rewrite Hv. reflexivity.
Qed.

Lemma lookup_none_keys {A B : Type} (l : list (positive * A))
    (l' : list (positive * B)) (k : positive) :
  map fst l' = map fst l -> lookup k l = None -> lookup k l' = None.
Proof.
  revert l'. induction l as [ | [k0 a] l IH]; intros [ | [k0' b] l'] Hk Hl;
    try discriminate Hk; [ reflexivity | ].
  injection Hk as -> Hk. rewrite lookup_cons in Hl. rewrite lookup_cons.
  destruct (Pos.eqb k k0); [ discriminate Hl | exact (IH l' Hk Hl) ].
Qed.

Lemma bindings_eval_rel (bs bs' : list static_binding) :
  rr_bindings M CH CS RCH W REM bs bs' ->
  ∀ (vars vars' : env) (S S' S1 : static_env),
    env_rel_c M CH vars vars' -> statics_rel M CH CS RCH W REM S S' ->
    eval_static_bindings vars bs S = Some S1 ->
    ∃ S1' : static_env,
      eval_static_bindings vars' bs' S' = Some S1'
      ∧ statics_rel M CH CS RCH W REM S1 S1'.
Proof.
  induction 1 as [ | b bs bs' Hb _ IH | id bs bs' _ IH
                 | id c c' ci bs bs' Hframe Hbody _ IH];
    intros vars vars' S S' S1 Hvars (Hval & Hkeys & Hcode) Heval.
  - simpl in Heval. injection Heval as <-. exists S'.
    split; [ reflexivity | split; [ exact Hval | split; assumption ] ].
  - destruct b as [ | | sym [ | | tag fields | | | ]]; simpl in Hb;
      try discriminate Hb.
    simpl in Heval |- *. rewrite Hval.
    destruct (lookup sym (static_values S)); [ discriminate Heval | ].
    simpl in Heval |- *.
    rewrite <- (plain_simples_eval M vars vars' fields (proj1 Hvars) Hb).
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
        exists ci. split; assumption.
      * exact (Hcode id0 c0 Hc0).
Qed.

Lemma resolve_rel (D : list continuation) (conts conts' : cont_env)
    (k : continuation) :
  conts_rel M CH CS RCH W REM D conts conts' -> k ∉ D ->
  cont_rel M CH CS RCH W REM (resolve k conts) (resolve k conts').
Proof. intros H Hk. exact (H k Hk). Qed.

Lemma lookup_rec_closures (all hs : list (continuation * continuation_handler))
    (vars : env) (conts : cont_env) (k : continuation) :
  lookup k (map (fun '(k0, h) => (k0, Closure h vars conts all)) hs)
  = option_map (fun h => Closure h vars conts all) (lookup k hs).
Proof.
  induction hs as [ | [k0 h] hs IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k0); [ reflexivity | exact IH ].
Qed.

Lemma resolve_rec (hs : list (continuation * continuation_handler))
    (vars : env) (conts : cont_env) (k : continuation) :
  resolve k (rec_closures hs vars conts ++ conts)
  = match lookup k hs with
    | Some h => Closure h vars conts hs
    | None => resolve k conts
    end.
Proof.
  unfold resolve, rec_closures. rewrite lookup_app, lookup_rec_closures.
  destruct (lookup k hs); reflexivity.
Qed.

Lemma rr_handlers_lookup (D : list continuation)
    (hs hs' : list (continuation * continuation_handler)) (k : continuation) :
  rr_handlers M CH CS RCH W REM D hs hs' ->
  match lookup k hs, lookup k hs' with
  | Some (Handler ps md h), Some (Handler ps' md' h') =>
    ps' = ps ∧ md' = md ∧ forallb (plain_variable M) ps = true
    ∧ rr M CH CS RCH W REM D h h'
  | None, None => True
  | _, _ => False
  end.
Proof.
  induction 1 as [ | D k0 ps md h h' hs hs' Hps Hh _ IH]; [ exact I | ].
  rewrite !lookup_cons. destruct (Pos.eqb k k0); [ | exact IH ].
  split; [ reflexivity | split; [ reflexivity | split; assumption ] ].
Qed.

Lemma rec_closures_rel (D : list continuation)
    (handlers handlers' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  rr_handlers M CH CS RCH W REM (without (map fst handlers) D) handlers
    handlers' ->
  env_rel_c M CH vars vars' -> conts_rel M CH CS RCH W REM D conts conts' ->
  conts_rel M CH CS RCH W REM (without (map fst handlers) D)
    (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hh Hvars Hconts k Hk. rewrite !resolve_rec.
  pose proof (rr_handlers_lookup _ _ _ k Hh) as Hl.
  destruct (lookup k handlers) as [[ps md h] | ] eqn:E1,
    (lookup k handlers') as [[ps' md' h'] | ] eqn:E2; try contradiction.
  - destruct Hl as (-> & -> & Hps & Hhr).
    exact (cont_rel_closure M CH CS RCH W REM D ps md h h' vars vars' conts
             conts' handlers handlers' Hps Hhr Hvars Hconts Hh).
  - apply Hconts. intros HkD. apply Hk. apply in_without_iff.
    split; [ exact HkD | ]. intros Hin.
    exact (lookup_keys k handlers Hin E1).
Qed.

Lemma bind_params_rel (ps : list variable) :
  ∀ (vs : list value) (vars vars' hv : env),
    forallb (plain_variable M) ps = true -> env_rel_c M CH vars vars' ->
    bind_params ps vs vars = Some hv ->
    ∃ hv' : env, bind_params ps vs vars' = Some hv' ∧ env_rel_c M CH hv hv'.
Proof.
  induction ps as [ | p ps IH]; intros [ | v vs] vars vars' hv Hps Hvars Hb;
    try discriminate Hb.
  - simpl in Hb |- *. injection Hb as <-. exists vars'.
    split; [ reflexivity | exact Hvars ].
  - simpl in Hps, Hb |- *. apply andb_true_iff in Hps as [Hp Hps].
    exact (IH vs _ _ hv Hps (env_rel_c_bind vars vars' p v Hvars Hp) Hb).
Qed.

Lemma with_args_same (ap : apply_expr) :
  with_args ap (ap_args ap) (ap_args_arity ap) = ap.
Proof. destruct ap. reflexivity. Qed.

Lemma env_rel_2 (vars vars' : env) :
  env_rel M vars vars' -> CSP.env_rel2 M [] vars vars'.
Proof.
  intros [Hp Hf]. split; [ | exact Hf ].
  intros z Hz. apply Hp. unfold CSP.plain2 in Hz.
  change (among z []) with false in Hz. cbn [negb] in Hz.
  rewrite andb_true_r in Hz. exact Hz.
Qed.

Lemma env_rel_of_2 (vars vars' : env) :
  CSP.env_rel2 M [] vars vars' -> env_rel M vars vars'.
Proof.
  intros [Hp Hf]. split; [ | exact Hf ].
  intros z Hz. apply Hp. unfold CSP.plain2.
  change (among z []) with false. cbn [negb].
  rewrite andb_true_r. exact Hz.
Qed.

(* The parameters of the second run: those of the first, or variables of
   fields. *)
Lemma params_rel2_vars (sh : list CSP.pshape)
    (pk pk' : list (variable * flambda_kind)) :
  CSP.params_rel2 M [] sh pk pk' ->
  ∀ z : variable, z ∈ map fst pk' ->
    z ∈ map fst pk ∨ among z (field_variables M) = true.
Proof.
  induction 1 as [ | sh p kind ps ps' _ _ IH | sh p kind ps ps' _ _ _ _ IH
                 | sh p fields ps ps' Hp _ _ IH];
    intros z Hz; cbn [map] in Hz |- *.
  - destruct Hz.
  - destruct Hz as [<- | Hz]; [ left; left; reflexivity | ].
    destruct (IH z Hz) as [H | H]; [ left; right; exact H | right; exact H ].
  - destruct Hz as [<- | Hz]; [ left; left; reflexivity | ].
    destruct (IH z Hz) as [H | H]; [ left; right; exact H | right; exact H ].
  - rewrite map_app in Hz. apply in_app_or in Hz as [Hz | Hz].
    + right. apply among_In. exact (field_variables_in M p fields z Hp Hz).
    + destruct (IH z Hz) as [H | H]; [ left; right; exact H | right; exact H ].
Qed.

(* Values related at the shapes of parameters, given to the parameters
   of the two runs, none changed, bind them as [env_rel_c] asks. *)
Lemma bind_vrel_c (sh : list CSP.pshape) (vs vs' : list value)
    (pk pk' : list (variable * flambda_kind)) (cv cv' hv : env) :
  CSP.vrel sh vs vs' -> CSP.params_rel2 M [] sh pk pk' ->
  env_rel_c M CH cv cv' -> (∀ p : variable, p ∈ map fst pk -> CH p = None) ->
  bind_params (map fst pk) vs cv = Some hv ->
  ∃ hv' : env,
    bind_params (map fst pk') vs' cv' = Some hv' ∧ env_rel_c M CH hv hv'.
Proof.
  intros Hv Hpk [Hrel Hch] Hnc Hb.
  destruct (CSP.bind_vrel M M_distinct [] sh vs vs' Hv pk pk' cv cv' hv Hpk
              (env_rel_2 cv cv' Hrel) Hb)
    as (hv' & Hb' & Hrel').
  exists hv'. split; [ exact Hb' | ].
  split; [ exact (env_rel_of_2 hv hv' Hrel') | ].
  intros x r w Hx Hl.
  assert (Hn : x ∉ map fst pk)
    by (intros Hin; rewrite (Hnc x Hin) in Hx; discriminate Hx).
  assert (Hn' : x ∉ map fst pk').
  { intros Hin. destruct (params_rel2_vars sh pk pk' Hpk x Hin) as [H | H];
      [ exact (Hn H) | ].
    rewrite (CH_fresh x r Hx) in H. discriminate H. }
  rewrite (CSP.bind_params_other _ vs cv hv x Hb Hn) in Hl.
  rewrite (CSP.bind_params_other _ vs' cv' hv' x Hb' Hn').
  exact (Hch x r w Hx Hl).
Qed.

Lemma params_frame_plain (ci : code_info) (c c' : code) :
  plain_shapes (ci_params ci) = true -> params_frame M CH ci c c' ->
  code_params c' = code_params c
  ∧ code_params_arity c' = code_params_arity c
  ∧ forallb (plain_variable M) (code_params c) = true
  ∧ code_params_subkinds c' = code_params_subkinds c.
Proof. intros Hp H. unfold params_frame in H. rewrite Hp in H. exact H. Qed.

Lemma env_rel_c_cw (vars vars' : env) :
  env_rel_c M CH vars vars' -> env_rel_cw M CH W vars vars'.
Proof.
  intros [[Hplain _] Hch]. split.
  - intros z _ Hz. exact (Hplain z Hz).
  - intros x r w _ Hr Hl. exact (Hch x r w Hr Hl).
Qed.

(* Within the wrappers of a call, of parameters of [W], the environments
   are related at the other names. *)
Lemma env_rel_cw_bind (cps wps : list variable) (vs : list value)
    (cvars wvars hv hv' : env) :
  env_rel_c M CH cvars wvars -> (∀ z : variable, z ∈ cps ++ wps -> W z) ->
  bind_params cps vs cvars = Some hv -> bind_params wps vs wvars = Some hv' ->
  env_rel_cw M CH W hv hv'.
Proof.
  intros [[Hplain _] Hch] HW Hb Hb'.
  assert (Hc : ∀ z : variable, ~ W z -> lookup z hv = lookup z cvars).
  { intros z Hz. apply (CSP.bind_params_other cps vs cvars hv z Hb).
    intros Hin. apply Hz, HW, in_or_app. left. exact Hin. }
  assert (Hw : ∀ z : variable, ~ W z -> lookup z hv' = lookup z wvars).
  { intros z Hz. apply (CSP.bind_params_other wps vs wvars hv' z Hb').
    intros Hin. apply Hz, HW, in_or_app. right. exact Hin. }
  split.
  - intros z Hz Hp. rewrite (Hc z Hz), (Hw z Hz). exact (Hplain z Hp).
  - intros x r w Hx Hr Hl. rewrite (Hc x Hx) in Hl. rewrite (Hw x Hx).
    exact (Hch x r w Hr Hl).
Qed.



(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Lemma named_plain_exec_c (S S' : static_env) (hp : heap) (rcd : read_record)
    (vars vars' : env) (d : named) :
  static_values S' = static_values S -> env_rel M vars vars' ->
  FlambdaReaperUnboxingRefinesProofs.named_plain M d = true ->
  exec_named S hp rcd vars d = exec_named S' hp rcd vars' d.
Proof.
  intros Hst Hrel Hd.
  destruct d as [s | p args | | | | | ]; simpl in Hd; try discriminate Hd.
  - simpl. rewrite (plain_simple_eval M vars vars' s Hrel Hd). reflexivity.
  - rewrite (exec_named_static_values S S' hp rcd vars p args)
      by (intros vs0 sym _ _; rewrite Hst; reflexivity).
    simpl. rewrite (plain_simples_eval M vars vars' args Hrel Hd).
    reflexivity.
Qed.

Lemma plain_slots_eval (vars vars' : env) (vs : list (value_slot * simple)) :
  env_rel M vars vars' ->
  forallb (fun ws => plain_simple M (snd ws)) vs = true ->
  eval_value_slots vars vs = eval_value_slots vars' vs.
Proof.
  intros Hrel. induction vs as [ | [w s] vs IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hs H].
  rewrite (plain_simple_eval M vars vars' s Hrel Hs), (IH H). reflexivity.
Qed.

Lemma slot_read_var (S : static_env) (vars : env) (d : named) (w : value_slot)
    (s : simple) (r0 : value) :
  slot_read d = Some (w, s) -> eval_named S vars d = Some r0 ->
  ∃ v : value, eval_simple vars s = Some v.
Proof.
  intros Hd Hev.
  destruct d as [ | | | | w0 s0 | fs w0 s0 | ]; try discriminate Hd;
    cbn [slot_read] in Hd; injection Hd as -> ->; cbn [eval_named] in Hev;
    destruct (eval_simple vars s) as [v | ]; try discriminate Hev;
    exists v; reflexivity.
Qed.

Lemma slot_read_not_heap (d : named) (w : value_slot) (s : simple) :
  slot_read d = Some (w, s) ->
  ∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args.
Proof. intros H q args ->. discriminate H. Qed.

Lemma slot_read_const (S : static_env) (vars : env) (d : named)
    (w : value_slot) (c : const) :
  slot_read d = Some (w, Const c) -> eval_named S vars d = None.
Proof.
  intros Hd. destruct d as [ | | | | w0 s0 | fs w0 s0 | ]; try discriminate Hd;
    cbn [slot_read] in Hd; injection Hd as -> ->; reflexivity.
Qed.

Lemma eval_proj_from (S : static_env) (vars : env) (cs : function_slot)
    (sl : value_slot) (y : variable) (v : value) :
  lookup y vars = Some v ->
  eval_named S vars (Project_value_slot_from cs sl (Var y))
  = project_value_slot S cs sl v.
Proof. intros H. cbn [eval_named eval_simple]. rewrite H. reflexivity. Qed.

Lemma exec_named_proj (S : static_env) (hp : heap) (rd : read_record)
    (vars : env) (cs : function_slot) (sl : value_slot) (y : simple) :
  exec_named S hp rd vars (Project_value_slot_from cs sl y)
  = match eval_named S vars (Project_value_slot_from cs sl y) with
    | Some v => Some (v, hp, rd)
    | None => None
    end.
Proof. reflexivity. Qed.

Lemma find_arm_in (i : Z) (arms : list (Z * apply_cont)) (ac : apply_cont) :
  find_arm i arms = Some ac -> ∃ d : Z, (d, ac) ∈ arms.
Proof.
  induction arms as [ | [d ac0] arms IH]; intros H; [ discriminate H | ].
  simpl in H. destruct (Z.eqb i d).
  - injection H as <-. exists d. left. reflexivity.
  - destruct (IH H) as (d' & Hin). exists d'. right. exact Hin.
Qed.

(* The values of a callee and of the [my_closure] of the code it calls:
   the same, or, for changed ones, closures related. *)
Definition class_rel (o : option closure_repr) (v v' : value) : Prop :=
  match o with
  | None => v' = v
  | Some r => crel_exact r v v'
  end.

Lemma closure_of_class (o : option closure_repr) (S S' : static_env)
    (v v' : value) (code : code_id) (sl : list (value_slot * value)) :
  class_rel o v v' -> static_values S' = static_values S ->
  closure_of S v = Some (code, sl) ->
  ∃ sl' : list (value_slot * value), closure_of S' v' = Some (code, sl').
Proof.
  intros Hrel Hst Hc. destruct o as [r | ]; cbn [class_rel] in Hrel.
  - destruct Hrel as [(decls & vs & vs' & -> & -> & Hdecls & _) _].
    cbn in Hc |- *. rewrite Hdecls.
    destruct (lookup (cr_current r) decls) as [code' | ];
      [ | discriminate Hc ].
    cbn in Hc |- *. injection Hc as <- _. exists vs'. reflexivity.
  - subst v'. exists sl. rewrite (closure_of_values S S' v Hst). exact Hc.
Qed.

Lemma callee_eval (ci : code_info) (s : simple) (vars vars' : env)
    (v : value) :
  env_rel_c M CH vars vars' -> callee_ok M CH ci s ->
  eval_simple vars s = Some v ->
  ∃ v' : value,
    eval_simple vars' s = Some v' ∧ class_rel (ci_my_closure ci) v v'.
Proof.
  intros [Hrel Hch] Hok Hv. unfold callee_ok in Hok.
  destruct (ci_my_closure ci) as [r | ]; cbn [class_rel].
  - destruct Hok as (f & -> & Hf). simpl in Hv |- *.
    exact (Hch f r v Hf Hv).
  - exists v. rewrite <- (plain_simple_eval M vars vars' s Hrel Hok).
    split; [ exact Hv | reflexivity ].
Qed.

Lemma my_closure_env (ci : code_info) (z : variable) (v v' : value) :
  my_closure_ok M CH ci z -> class_rel (ci_my_closure ci) v v' ->
  env_rel_c M CH [(z, v)] [(z, v')].
Proof.
  unfold my_closure_ok. destruct (ci_my_closure ci) as [r | ];
    cbn [class_rel]; intros Hz Hrel.
  - exact (env_rel_c_bind_changed [] [] z r v v' env_rel_c_empty Hz Hrel).
  - subst v'. exact (env_rel_c_bind [] [] z v env_rel_c_empty Hz).
Qed.

Lemma without_nil (l : list continuation) : without [] l = l.
Proof.
  unfold without. induction l as [ | a l IH]; [ reflexivity | ].
  cbn [filter]. rewrite IH. reflexivity.
Qed.

Lemma with_continuation_same (ap : apply_expr) :
  with_continuation ap (ap_continuation ap) = ap.
Proof. destruct ap. reflexivity. Qed.

Lemma resolve_other (k k0 : continuation) (c : cont_value)
    (conts : cont_env) :
  k <> k0 -> resolve k ((k0, c) :: conts) = resolve k conts.
Proof.
  intros Hk. unfold resolve. rewrite lookup_cons.
  rewrite (proj2 (Pos.eqb_neq k k0) Hk). reflexivity.
Qed.

Lemma run_two (t t1 t2 : state) :
  step t = Step t1 -> step t1 = Step t2 -> run 2 t = Step t2.
Proof.
  intros H1 H2.
  change (run 2 t) with (match step t with Step s' => run 1 s' | r => r end).
  rewrite H1.
  change (run 1 t1) with (match step t1 with Step s' => run 0 s' | r => r end).
  rewrite H2. reflexivity.
Qed.

(* A full call of the code [id] in both runs, its [my_closure] given
   values of its class, the arguments of the second run those of the
   first at the shapes of the parameters of the code: both runs enter
   related code, the frames given as they are. *)
Lemma call_code_entered_u (id : code_id) (ci : code_info) (cc cc' : code)
    (c1 c2 : value) (vs vs' : list value) (aa aa' rta : arity)
    (r r' x x' : cont_value) (S S' : static_env) (rcd : read_record) :
  code_frame M CH CS id cc cc' ci ->
  rr M CH CS RCH W REM (code_dead ci cc) (code_body cc) (code_body cc') ->
  class_rel (ci_my_closure ci) c1 c2 -> static_values S' = static_values S ->
  (if plain_shapes (ci_params ci) then vs' = vs ∧ aa' = aa
   else
     CSP.vrel (ci_params ci) vs vs' ∧ aa' = ci_params_arity ci
     ∧ CSP.shape_kinds (ci_params ci) (unarize aa)
       = unarize (ci_params_arity ci)) ->
  match call_code c1 cc vs aa rta r x S empty_heap rcd with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ hv hv' : env,
      s1 = State (code_body cc) hv
             [(code_return_continuation cc, r);
              (code_exn_continuation cc, x)]
             S empty_heap rcd
      ∧ call_code c2 cc' vs' aa' rta r' x' S' empty_heap rcd
        = Step (State (code_body cc') hv'
                  [(code_return_continuation cc, r');
                   (code_exn_continuation cc, x')]
                  S' empty_heap rcd)
      ∧ env_rel_c M CH hv hv'
  end.
Proof.
  intros Hframe Hcbody Hcl Hsv Hargs.
  destruct Hframe as (_ & Hpf & Hmc & Hret & Hexn & Hra & _ & _ & _ & Hmcok).
  unfold call_code.
  destruct (binds_statics (code_body cc)) eqn:Hbs; [ exact I | ].
  assert (Hbs' : binds_statics (code_body cc') = false).
  { destruct (binds_statics (code_body cc')) eqn:E; [ | reflexivity ].
    rewrite (proj2 (proj1 rr_shape _ _ _ Hcbody) E) in Hbs.
    discriminate Hbs. }
  rewrite Hbs', Hra.
  unfold params_frame in Hpf.
  destruct (plain_shapes (ci_params ci)).
  - destruct Hargs as [-> ->]. destruct Hpf as (Hps & Hpa & Hplain & Hsk').
    rewrite Hpa.
    match goal with
    | |- context [if negb ?t then Undefined else _] =>
      destruct t eqn:Htest
    end; cbn [negb]; [ | exact I ].
    rewrite Hsk', (values_of_subkinds_same_values S' S vs _ Hsv).
    destruct (values_of_subkinds S vs (code_params_subkinds cc)) eqn:Hsk;
      cbn [negb]; [ | exact I ].
    destruct (bind_params (code_params cc) vs [(code_my_closure cc, c1)])
      as [hv | ] eqn:Hb; [ | exact I ].
    destruct (bind_params_rel (code_params cc) vs [(code_my_closure cc, c1)]
                [(code_my_closure cc, c2)] hv Hplain
                (my_closure_env ci (code_my_closure cc) c1 c2 Hmcok Hcl) Hb)
      as (hv' & Hb' & Hrel).
    exists hv, hv'. split; [ reflexivity | split; [ | exact Hrel ] ].
    rewrite Hps, Hmc, Hb', Hret, Hexn. reflexivity.
  - destruct Hargs as (Hv & -> & Hsh).
    destruct Hpf as (Hnd & Hnc & (pk & pk' & Hpk & Hpk' & Hrel2) & Hpa & _
                     & Hka & Hsks).
    rewrite Hpa.
    destruct (kinds_eqb (map kind_of_value vs) (unarize aa)) eqn:Hk;
      cbn [andb negb]; [ | exact I ].
    destruct (kinds_per_parameter_eqb (unarize_per_parameter aa)
                (unarize_per_parameter (code_params_arity cc))) eqn:Hkp;
      cbn [andb negb]; [ | exact I ].
    destruct (arity_eqb rta (code_result_arity cc)) eqn:Hra2;
      cbn [andb negb]; [ | exact I ].
    assert (Hk' : kinds_eqb (map kind_of_value vs')
                    (unarize (ci_params_arity ci)) = true).
    { apply kinds_eqb_eq. apply kinds_eqb_eq in Hk.
      rewrite (CSP.vrel_shape_kinds _ _ _ Hv Hnd), Hk. exact Hsh. }
    assert (Hkp' : kinds_per_parameter_eqb
                     (unarize_per_parameter (ci_params_arity ci))
                     (unarize_per_parameter (ci_params_arity ci)) = true)
      by (apply kinds_per_parameter_eqb_eq; reflexivity).
    rewrite Hk', Hkp'. cbn [andb negb].
    destruct (values_of_subkinds S vs (code_params_subkinds cc)) eqn:Hsk;
      cbn [negb]; [ | exact I ].
    rewrite (CSP.sks_follow2_statics _ _ _ S S' vs vs' Hsks Hsv Hv Hsk).
    cbn [negb].
    rewrite Hpk in Hnc |- *.
    destruct (bind_params (map fst pk) vs [(code_my_closure cc, c1)])
      as [hv | ] eqn:Hb; [ | exact I ].
    destruct (bind_vrel_c (ci_params ci) vs vs' pk pk'
                [(code_my_closure cc, c1)] [(code_my_closure cc, c2)] hv Hv
                Hrel2 (my_closure_env ci (code_my_closure cc) c1 c2 Hmcok Hcl)
                Hnc Hb)
      as (hv' & Hb' & Hrel).
    exists hv, hv'. split; [ reflexivity | split; [ | exact Hrel ] ].
    rewrite Hpk', Hmc, Hb', Hret, Hexn. reflexivity.
Qed.

(* The same, of code none of whose parameters is unboxed, with the same
   arguments in both runs. *)
Lemma call_code_entered (id : code_id) (ci : code_info) (cc cc' : code)
    (c1 c2 : value) (vs : list value) (aa rta : arity)
    (r r' x x' : cont_value) (S S' : static_env) (rcd : read_record) :
  code_frame M CH CS id cc cc' ci ->
  rr M CH CS RCH W REM (code_dead ci cc) (code_body cc) (code_body cc') ->
  class_rel (ci_my_closure ci) c1 c2 -> static_values S' = static_values S ->
  plain_shapes (ci_params ci) = true ->
  match call_code c1 cc vs aa rta r x S empty_heap rcd with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ hv hv' : env,
      s1 = State (code_body cc) hv
             [(code_return_continuation cc, r);
              (code_exn_continuation cc, x)]
             S empty_heap rcd
      ∧ call_code c2 cc' vs aa rta r' x' S' empty_heap rcd
        = Step (State (code_body cc') hv'
                  [(code_return_continuation cc, r');
                   (code_exn_continuation cc, x')]
                  S' empty_heap rcd)
      ∧ env_rel_c M CH hv hv'
  end.
Proof.
  intros Hframe Hcbody Hcl Hsv Hp.
  pose proof (call_code_entered_u id ci cc cc' c1 c2 vs vs aa aa rta r r' x x'
                S S' rcd Hframe Hcbody Hcl Hsv) as H.
  rewrite Hp in H. exact (H (conj eq_refl eq_refl)).
Qed.

(* A direct call [ap] of the code [id] in the first run, and [ap2] in the
   second, of the same callee, the arguments those of [ap] at the shapes
   of the parameters of the code ([args_frame]): both runs enter related
   code, the frames of their return continuations unrelated. *)
Lemma call_entered_u (ap ap2 : apply_expr) (id : code_id) (ci : code_info)
    (vars vars' : env) (conts conts' : cont_env) (S S' : static_env)
    (rcd : read_record) :
  ap_call_kind ap = Direct id -> CS id = Some ci ->
  callee_ok M CH ci (ap_callee ap) ->
  forallb (plain_simple M) (exn_extra_simples ap) = true ->
  length (ap_args_arity ap) = ci_nparams ci ->
  ap_callee ap2 = ap_callee ap -> ap_call_kind ap2 = ap_call_kind ap ->
  ap_exn_continuation ap2 = ap_exn_continuation ap ->
  ap_exn_extra_args ap2 = ap_exn_extra_args ap ->
  ap_return_arity ap2 = ap_return_arity ap ->
  args_frame M ci ap (ap_args ap2) (ap_args_arity ap2) ->
  env_rel_c M CH vars vars' -> statics_rel M CH CS RCH W REM S S' ->
  match step (State (Apply ap) vars conts S empty_heap rcd) with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ (cc cc' : code) (hv hv' : env) (extra : list value),
      s1 = State (code_body cc) hv
             [(code_return_continuation cc,
               resolve (ap_continuation ap) conts);
              (code_exn_continuation cc,
               with_extra_args (resolve (ap_exn_continuation ap) conts)
                 extra)]
             S empty_heap rcd
      ∧ step (State (Apply ap2) vars' conts' S' empty_heap rcd)
        = Step (State (code_body cc') hv'
                  [(code_return_continuation cc,
                    resolve (ap_continuation ap2) conts');
                   (code_exn_continuation cc,
                    with_extra_args (resolve (ap_exn_continuation ap) conts')
                      extra)]
                  S' empty_heap rcd)
      ∧ env_rel_c M CH hv hv'
      ∧ rr M CH CS RCH W REM (code_dead ci cc) (code_body cc) (code_body cc')
  end.
Proof.
  intros Hkind Hci Hcallee Hextra Hlen Hc2e Hk2e Hexn2 Hex2 Hrta2 Hargs Hvars
    Hst.
  rewrite step_apply.
  destruct (eval_simple vars (ap_callee ap)) as [c1 | ] eqn:Hc1;
    [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
    [ | exact I ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ] eqn:Hex;
    [ | exact I ].
  destruct (callee_eval ci (ap_callee ap) vars vars' c1 Hvars Hcallee Hc1)
    as (c2 & Hc2 & Hcl).
  unfold call_function, run_call, shape_of_call. rewrite Hkind.
  cbn beta iota. unfold shape_of_closure_call.
  destruct (closure_of S c1) as [[called sl1] | ] eqn:Hcl1; [ | exact I ].
  destruct (closure_of_class _ S S' c1 c2 called sl1 Hcl (proj1 Hst) Hcl1)
    as (sl2 & Hcl2).
  destruct (Pos.eqb called id) eqn:Hid; [ | exact I ].
  apply Pos.eqb_eq in Hid. subst called.
  destruct (lookup id (static_code S)) as [cc | ] eqn:Hcc; [ | exact I ].
  destruct (proj2 (proj2 Hst) id cc Hcc)
    as (cc' & Hcc' & ci' & Hframe & Hcbody).
  pose proof Hframe as (Hci' & Hpf & _ & _ & _ & _ & Htup & Htup' & Hlenc
                        & _).
  rewrite Hci in Hci'. injection Hci' as <-.
  assert (Hvs' : ∃ vs' : list value,
             eval_simples vars' (ap_args ap2) = Some vs'
             ∧ length (ap_args_arity ap2) = length (code_params_arity cc')
             ∧ if plain_shapes (ci_params ci) then
                 vs' = vs ∧ ap_args_arity ap2 = ap_args_arity ap
               else
                 CSP.vrel (ci_params ci) vs vs'
                 ∧ ap_args_arity ap2 = ci_params_arity ci
                 ∧ CSP.shape_kinds (ci_params ci) (unarize (ap_args_arity ap))
                   = unarize (ci_params_arity ci)).
  { unfold args_frame in Hargs. unfold params_frame in Hpf.
    destruct (plain_shapes (ci_params ci)).
    - destruct Hargs as (-> & -> & Hpl). destruct Hpf as (_ & Hpa & _).
      exists vs.
      split; [ rewrite <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hpl);
               exact Hvs | ].
      split; [ rewrite Hpa, Hlen, Hlenc; reflexivity | ].
      split; reflexivity.
    - destruct Hargs as (Hrel2 & Haa & Hsh).
      destruct Hpf as (_ & _ & _ & Hpa & Hlpa & _).
      destruct (CSP.args_vrel M [] _ _ _ Hrel2 vars vars' vs
                  (env_rel_2 _ _ (proj1 Hvars)) Hvs)
        as (vs' & Hvs' & Hv).
      exists vs'. split; [ exact Hvs' | ].
      split; [ rewrite Haa, Hpa; reflexivity | ].
      split; [ exact Hv | split; [ exact Haa | exact Hsh ] ]. }
  destruct Hvs' as (vs' & Hvs' & Hlen2 & Hshape).
  unfold arguments_for_code. rewrite Htup. cbn beta iota.
  unfold shape_of_code_call. rewrite Hlen, <- Hlenc, Nat.eqb_refl.
  cbn beta iota.
  pose proof (call_code_entered_u id ci cc cc' c1 c2 vs vs' (ap_args_arity ap)
                (ap_args_arity ap2) (ap_return_arity ap)
                (resolve (ap_continuation ap) conts)
                (resolve (ap_continuation ap2) conts')
                (with_extra_args (resolve (ap_exn_continuation ap) conts)
                   extra)
                (with_extra_args (resolve (ap_exn_continuation ap) conts')
                   extra)
                S S' rcd Hframe Hcbody Hcl (proj1 Hst) Hshape)
    as Hentered.
  destruct (call_code c1 cc vs (ap_args_arity ap) (ap_return_arity ap)
              (resolve (ap_continuation ap) conts)
              (with_extra_args (resolve (ap_exn_continuation ap) conts)
                 extra)
              S empty_heap rcd)
    as [s1 | | ]; [ | contradiction | exact I ].
  destruct Hentered as (hv & hv' & -> & Hcall' & Hrel).
  exists cc, cc', hv, hv', extra. split; [ reflexivity | ].
  split; [ | split; [ exact Hrel | exact Hcbody ] ].
  rewrite step_apply, Hc2e, Hk2e, Hexn2, Hrta2, Hc2, Hvs'.
  change (exn_extra_simples ap2) with (map fst (ap_exn_extra_args ap2)).
  rewrite Hex2.
  change (map fst (ap_exn_extra_args ap)) with (exn_extra_simples ap).
  rewrite <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hextra), Hex.
  unfold call_function, run_call, shape_of_call. rewrite Hkind.
  cbn beta iota. unfold shape_of_closure_call.
  rewrite Hcl2, Pos.eqb_refl, Hcc'.
  unfold arguments_for_code. rewrite Htup'. cbn beta iota.
  unfold shape_of_code_call. rewrite Hlen2, Nat.eqb_refl.
  cbn beta iota. exact Hcall'.
Qed.

(* A call of [call_ok_c], the second run returning to [k2]: both runs
   enter related code, the frames of their return continuations
   unrelated. *)
Lemma call_entered (ap : apply_expr) (k2 : continuation) (id : code_id)
    (ci : code_info) (vars vars' : env) (conts conts' : cont_env)
    (S S' : static_env) (rcd : read_record) :
  call_ok_c M CH CS ap id ci ->
  env_rel_c M CH vars vars' -> statics_rel M CH CS RCH W REM S S' ->
  match step (State (Apply ap) vars conts S empty_heap rcd) with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ (cc cc' : code) (hv hv' : env) (extra : list value),
      s1 = State (code_body cc) hv
             [(code_return_continuation cc,
               resolve (ap_continuation ap) conts);
              (code_exn_continuation cc,
               with_extra_args (resolve (ap_exn_continuation ap) conts)
                 extra)]
             S empty_heap rcd
      ∧ step (State (Apply (with_continuation ap k2)) vars' conts' S'
                empty_heap rcd)
        = Step (State (code_body cc') hv'
                  [(code_return_continuation cc, resolve k2 conts');
                   (code_exn_continuation cc,
                    with_extra_args (resolve (ap_exn_continuation ap) conts')
                      extra)]
                  S' empty_heap rcd)
      ∧ env_rel_c M CH hv hv'
      ∧ rr M CH CS RCH W REM (code_dead ci cc) (code_body cc) (code_body cc')
  end.
Proof.
  intros (Hkind & Hci & Hcallee & Hargs & Hextra & Hlen & Hpl) Hvars Hst.
  refine (call_entered_u ap (with_continuation ap k2) id ci vars vars' conts
            conts' S S' rcd Hkind Hci Hcallee Hextra Hlen eq_refl eq_refl
            eq_refl eq_refl eq_refl _ Hvars Hst).
  unfold args_frame. rewrite Hpl.
  split; [ reflexivity | split; [ reflexivity | exact Hargs ] ].
Qed.

(* The frames of code entered: those of its exception continuation
   related, and those of its return continuation if it may return. *)
Lemma code_conts_rel (ci : code_info) (cc : code) (r r' x x' : cont_value) :
  (ci_never ci = false -> cont_rel M CH CS RCH W REM r r') ->
  cont_rel M CH CS RCH W REM x x' ->
  conts_rel M CH CS RCH W REM (code_dead ci cc)
    [(code_return_continuation cc, r); (code_exn_continuation cc, x)]
    [(code_return_continuation cc, r'); (code_exn_continuation cc, x')].
Proof.
  intros Hr Hx k Hk. unfold resolve. rewrite !lookup_cons.
  destruct (Pos.eqb k (code_return_continuation cc)) eqn:E.
  - apply Pos.eqb_eq in E. subst k. apply Hr. unfold code_dead in Hk.
    destruct (ci_never ci); [ | reflexivity ].
    exfalso. apply Hk. left. reflexivity.
  - destruct (Pos.eqb k (code_exn_continuation cc)); [ exact Hx | ].
    constructor.
Qed.

(* The exception continuation of a call with extra arguments. *)
Lemma cont_rel_with_extra (c c' : cont_value) (extra : list value) :
  cont_rel M CH CS RCH W REM c c' ->
  cont_rel M CH CS RCH W REM (with_extra_args c extra)
    (with_extra_args c' extra).
Proof.
  intros H. destruct extra; [ exact H | ].
  exact (cont_rel_extra M CH CS RCH W REM _ _ _ H).
Qed.

(* The first run at a call of a variable [f]: a partial application that
   [f] holds is made by the machine, with fewer groups of arguments than
   its code has parameters, and the code of the closure that [f] holds, or
   of the closure that a call of it calls, is in [RCH] of [f]. *)
Definition flow_ok (s : state) : Prop :=
  ∀ (ap : apply_expr) (f : variable) (v : value),
    state_expr s = Apply ap -> ap_callee ap = Var f ->
    lookup f (state_vars s) = Some v ->
    (∀ (cl : value) (applied : list value) (aa : arity),
       v = Value_partial_application cl applied aa ->
       PGP.pa_bound (state_statics s) cl aa)
    ∧ ∀ (id : code_id) (slots : list (value_slot * value)),
        (closure_of (state_statics s) v = Some (id, slots) -> RCH true f id)
        ∧ (closure_of (state_statics s) (call_closure v) = Some (id, slots)
           -> RCH false f id).

(* The closure of a partial application that the machine made, called
   with one group of arguments more, is not over-applied
   (FlambdaReaperUnboxingRefinesCallsSimProofs.partial_one_no_over, of the
   bound of the partial application alone). *)
Lemma partial_one_no_over_b (S : static_env) (cl : value)
    (applied : list value) (aa : arity) (vs : list value)
    (g : arity_component) (rta : arity) (cl0 : value) (c0 : code)
    (ws rest : list value) (aa0 ra1 : arity) :
  PGP.pa_bound S cl aa ->
  (∀ (id : code_id) (c : code),
     lookup id (static_code S) = Some c -> code_is_tupled c = false) ->
  shape_of_closure_call S cl Indirect_unknown_arity (applied ++ vs)
    (aa ++ [g]) rta
  <> Some (Over_call cl0 c0 ws rest aa0 ra1).
Proof.
  intros Hb Hcodes H.
  destruct Hb as (called' & slots' & c' & Hcl' & Hc' & Hlt').
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ H)
    as (called & slots & c & vs' & aa' & Hcl & Hc & _ & Hargs & Hcall).
  rewrite Hcl' in Hcl. injection Hcl as <- <-.
  rewrite Hc' in Hc. injection Hc as <-.
  pose proof (Hcodes called' c' Hc') as Htup.
  unfold arguments_for_code in Hargs. rewrite Htup in Hargs.
  injection Hargs as <- <-.
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[_ E] | [(Hlt & _ & _ & _) | (_ & _ & _ & _ & E)]];
    try discriminate E.
  rewrite length_app in Hlt. cbn [length] in Hlt. lia.
Qed.

(* The same closure called in both runs: the call is of the same shape,
   of related code. *)
Lemma closure_shape_rel (S S' : static_env) (cl : value) (kind : call_kind)
    (vs : list value) (ra rta : arity) :
  (∀ (id : code_id) (sl : list (value_slot * value)) (ci : code_info),
     closure_of S cl = Some (id, sl) -> CS id = Some ci ->
     plain_shapes (ci_params ci) = true) ->
  statics_rel M CH CS RCH W REM S S' ->
  match shape_of_closure_call S cl kind vs ra rta with
  | Some (Full_call cl0 c vs0 aa) =>
    ∃ (called : code_id) (sl : list (value_slot * value)) (c' : code)
      (ci : code_info),
      closure_of S cl = Some (called, sl) ∧ cl0 = cl
      ∧ code_frame M CH CS called c c' ci
      ∧ rr M CH CS RCH W REM (code_dead ci c) (code_body c) (code_body c')
      ∧ shape_of_closure_call S' cl kind vs ra rta
        = Some (Full_call cl c' vs0 aa)
  | Some (Partial_call p) =>
    shape_of_closure_call S' cl kind vs ra rta = Some (Partial_call p)
  | Some (Over_call cl0 c ws rest aa ra1) =>
    ∃ (called : code_id) (sl : list (value_slot * value)) (c' : code)
      (ci : code_info),
      closure_of S cl = Some (called, sl) ∧ cl0 = cl
      ∧ code_frame M CH CS called c c' ci
      ∧ rr M CH CS RCH W REM (code_dead ci c) (code_body c) (code_body c')
      ∧ shape_of_closure_call S' cl kind vs ra rta
        = Some (Over_call cl c' ws rest aa ra1)
  | None => True
  end.
Proof.
  intros Hpl Hst. unfold shape_of_closure_call. cbv zeta.
  rewrite (closure_of_values S S' cl (proj1 Hst)).
  destruct (closure_of S cl) as [[called sl] | ] eqn:Hcl; [ | exact I ].
  destruct (lookup called (static_code S)) as [c | ] eqn:Hc.
  2: { destruct (match kind with
                 | Direct named => Pos.eqb called named
                 | Indirect_unknown_arity => true
                 | Indirect_known_arity code_ids =>
                   (code_ids_allow code_ids called
                    && known_arity_full S called ra)%bool
                 end); exact I. }
  destruct (proj2 (proj2 Hst) called c Hc)
    as (c' & Hc' & ci & Hframe & Hbody).
  pose proof Hframe as (_ & Hpf & _ & _ & _ & Hra & Htup & Htup' & _).
  pose proof (proj1 (proj2 (params_frame_plain ci c c'
                              (Hpl called sl ci eq_refl (proj1 Hframe)) Hpf)))
    as Hpa.
  assert (Hka : known_arity_full S' called ra = known_arity_full S called ra)
    by (unfold known_arity_full; rewrite Hc, Hc', Htup, Htup', Hpa;
        reflexivity).
  rewrite Hka, Hc'.
  destruct (match kind with
            | Direct named => Pos.eqb called named
            | Indirect_unknown_arity => true
            | Indirect_known_arity code_ids =>
              (code_ids_allow code_ids called
               && known_arity_full S called ra)%bool
            end); [ | exact I ].
  unfold arguments_for_code. rewrite Htup, Htup'. cbn beta iota.
  unfold shape_of_code_call. rewrite Hpa, Hra, Htup'. cbv zeta.
  destruct (Nat.eqb (length ra) (length (code_params_arity c))).
  - exists called, sl, c', ci.
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ exact Hframe | split; [ exact Hbody | reflexivity ] ].
  - destruct (kinds_eqb (map kind_of_value vs) (unarize ra)
              && partial_or_over_application ra rta (code_params_arity c)
                   (code_result_arity c))%bool;
      [ | exact I ].
    destruct (Nat.ltb (length (code_params_arity c)) (length ra)).
    + exists called, sl, c', ci.
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ exact Hframe | split; [ exact Hbody | reflexivity ] ].
    + rewrite Htup. reflexivity.
Qed.

(* A call of a closure in both runs, of code whose [my_closure] is plain:
   both runs enter related code with the frames given, through what is
   left of an over-application if the call over-applies (of code of [REM]
   whose [my_closure] is plain), or give the same partial application to
   their frames, which answer it. *)
Lemma closure_call_answered (cl : value) (kind : call_kind)
    (vs : list value) (ra rta : arity) (r r' x x' : cont_value)
    (S S' : static_env) (rcd : read_record) :
  (∀ (id : code_id) (sl : list (value_slot * value)) (ci : code_info),
     closure_of S cl = Some (id, sl) -> CS id = Some ci -> ci_plain ci) ->
  (∀ (cl0 : value) (c0 : code) (ws rest : list value) (aa ra1 : arity),
     shape_of_closure_call S cl kind vs ra rta
     = Some (Over_call cl0 c0 ws rest aa ra1) ->
     ∀ (id : code_id) (ci : code_info),
       REM id -> CS id = Some ci -> ci_plain ci) ->
  cont_rel M CH CS RCH W REM r r' -> cont_rel M CH CS RCH W REM x x' ->
  statics_rel M CH CS RCH W REM S S' ->
  (∀ p : value,
     shape_of_closure_call S cl kind vs ra rta = Some (Partial_call p) ->
     match apply_cont_value r [p] S empty_heap rcd with
     | Undefined => True
     | Exit k ws hp rd1 =>
       apply_cont_value r' [p] S' empty_heap rcd = Exit k ws hp rd1
     | Step s1 =>
       ∃ t1 : state,
         apply_cont_value r' [p] S' empty_heap rcd = Step t1
         ∧ related M CH CS RCH W REM s1 t1
     end) ->
  match
    run_call (fun p => apply_cont_value r [p] S empty_heap rcd)
      (shape_of_closure_call S cl kind vs ra rta) rta r x S empty_heap rcd
  with
  | Undefined => True
  | Exit k ws hp rd1 =>
    run_call (fun p => apply_cont_value r' [p] S' empty_heap rcd)
      (shape_of_closure_call S' cl kind vs ra rta) rta r' x' S' empty_heap
      rcd
    = Exit k ws hp rd1
  | Step s1 =>
    ∃ t1 : state,
      run_call (fun p => apply_cont_value r' [p] S' empty_heap rcd)
        (shape_of_closure_call S' cl kind vs ra rta) rta r' x' S'
        empty_heap rcd
      = Step t1
      ∧ related M CH CS RCH W REM s1 t1
  end.
Proof.
  intros Hmy Hover Hr Hx Hst Hpart.
  pose proof (closure_shape_rel S S' cl kind vs ra rta
                (fun id sl ci H1 H2 => proj2 (Hmy id sl ci H1 H2)) Hst)
    as Hsh.
  destruct (shape_of_closure_call S cl kind vs ra rta)
    as [[cl0 c vs0 aa | cl0 c ws rest aa ra1 | p] | ] eqn:E;
    [ | | | exact I ].
  - destruct Hsh as (called & sl & c' & ci & Hcl & -> & Hframe & Hbody & Hsh).
    rewrite Hsh. cbn [run_call].
    assert (Hclass : class_rel (ci_my_closure ci) cl cl)
      by (rewrite (proj1 (Hmy called sl ci Hcl (proj1 Hframe))); reflexivity).
    pose proof (call_code_entered called ci c c' cl cl vs0 aa rta r r' x x'
                  S S' rcd Hframe Hbody Hclass (proj1 Hst)
                  (proj2 (Hmy called sl ci Hcl (proj1 Hframe))))
      as Hentered.
    destruct (call_code cl c vs0 aa rta r x S empty_heap rcd) as [s1 | | ];
      [ | contradiction | exact I ].
    destruct Hentered as (hv & hv' & -> & Hcall & Hrel).
    eexists. split; [ exact Hcall | ].
    left. exists (code_dead ci c).
    split; [ exact Hbody | split; [ exact Hrel | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    apply code_conts_rel; [ intros _; exact Hr | exact Hx ].
  - destruct Hsh as (called & sl & c' & ci & Hcl & -> & Hframe & Hbody & Hsh).
    rewrite Hsh. cbn [run_call].
    pose proof Hframe as (_ & _ & _ & _ & _ & Hra & _).
    rewrite Hra.
    assert (Hclass : class_rel (ci_my_closure ci) cl cl)
      by (rewrite (proj1 (Hmy called sl ci Hcl (proj1 Hframe))); reflexivity).
    pose proof (call_code_entered called ci c c' cl cl ws aa
                  (code_result_arity c) (Over_application rest ra1 rta r x)
                  (Over_application rest ra1 rta r' x') x x' S S' rcd Hframe
                  Hbody Hclass (proj1 Hst)
                  (proj2 (Hmy called sl ci Hcl (proj1 Hframe))))
      as Hentered.
    destruct (call_code cl c ws aa (code_result_arity c)
                (Over_application rest ra1 rta r x) x S empty_heap rcd)
      as [s1 | | ]; [ | contradiction | exact I ].
    destruct Hentered as (hv & hv' & -> & Hcall & Hrel).
    eexists. split; [ exact Hcall | ].
    left. exists (code_dead ci c).
    split; [ exact Hbody | split; [ exact Hrel | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    apply code_conts_rel; [ intros _ | exact Hx ].
    apply cont_rel_over; [ exact Hr | exact Hx | ].
    exact (Hover _ _ _ _ _ _ eq_refl).
  - rewrite Hsh. cbn [run_call]. exact (Hpart p eq_refl).
Qed.

(* A continuation applied in both runs to the same values, what is left
   of an over-application calling code of [REM] ([PGP.cont_rem_ok]). *)
Lemma cont_answered (c c' : cont_value) (vs : list value)
    (S S' : static_env) (rd : read_record) :
  cont_rel M CH CS RCH W REM c c' -> statics_rel M CH CS RCH W REM S S' ->
  PGP.cont_rem_ok REM S c vs ->
  match apply_cont_value c vs S empty_heap rd with
  | Undefined => True
  | Exit k ws hp rd1 =>
    apply_cont_value c' vs S' empty_heap rd = Exit k ws hp rd1
  | Step s1 =>
    ∃ t1 : state,
      apply_cont_value c' vs S' empty_heap rd = Step t1
      ∧ related M CH CS RCH W REM s1 t1
  end.
Proof.
  intros Hc Hst. revert vs.
  induction Hc as [D ps md h h' vars vars' conts conts' rec rec' Hps Hh Hvars
                     Hconts _ Hrec | k | c c' extra Hc IH
                  | cps cmd k cargs cvars cconts wps wmd wargs wvars wconts
                      Hk _ Hvars HW Hwrap
                  | remaining ra rta c c' e e' Hc IHc He _ Hremp];
    intros vs Hrem; cbn [apply_cont_value];
    [ | reflexivity | exact (IH _ Hrem) | | ].
  3: { (* What is left of an over-application: both runs call the value
          they are given, of code of [REM], with the arguments left. *)
    destruct vs as [ | v [ | v' vs]]; cbn beta iota; [ exact I | | exact I ].
    cbn [PGP.cont_rem_ok] in Hrem. destruct Hrem as [Hcode Hshape].
    unfold shape_of_call in Hshape |- *.
    destruct (partial_application_of v) as [[[cl applied] aa] | ] eqn:Hpv.
    - destruct ra as [ | g ra']; [ exact I | ].
      destruct (kinds_eqb (map kind_of_value remaining) (unarize (g :: ra')));
        [ | exact I ].
      apply closure_call_answered; try assumption.
      + intros id sl ci Hcl Hci. apply (Hremp id ci); [ | exact Hci ].
        apply (Hcode id sl). unfold PGP.closure_called. rewrite Hpv.
        exact Hcl.
      + intros _ _ _ _ _ _ _. exact Hremp.
      + intros p Hp. rewrite Hp in Hshape. exact (IHc [p] Hshape).
    - apply closure_call_answered; try assumption.
      + intros id sl ci Hcl Hci. apply (Hremp id ci); [ | exact Hci ].
        apply (Hcode id sl). unfold PGP.closure_called. rewrite Hpv.
        exact Hcl.
      + intros _ _ _ _ _ _ _. exact Hremp.
      + intros p Hp. rewrite Hp in Hshape. exact (IHc [p] Hshape). }
  2: { (* The wrappers: both runs come within them. *)
    destruct Hwrap as (Hkinds & Hsk & Hbind). rewrite Hkinds.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds cmd));
      [ | exact I ].
    destruct (values_of_subkinds S vs (hm_params_subkinds cmd)) eqn:Hs;
      [ | exact I ].
    rewrite (values_of_subkinds_same_values S' S vs _ (proj1 Hst)),
      (Hsk S vs Hs).
    destruct (bind_params cps vs cvars) as [hv | ] eqn:Hb; [ | exact I ].
    destruct (Hbind vs cvars wvars hv Hb) as (hv' & Hb' & Hargs).
    rewrite Hb'. eexists. split; [ reflexivity | ].
    right. right.
    exists {| ac_continuation := k; ac_args := cargs;
              ac_trap_action := None |},
      {| ac_continuation := k; ac_args := wargs; ac_trap_action := None |}.
    cbn [state_expr state_vars state_conts state_statics state_heap
         state_reads rec_closures map app ac_continuation ac_args].
    split; [ reflexivity | split; [ reflexivity | split; [ reflexivity | ] ] ].
    split; [ exact Hk | split; [ exact Hargs | ] ].
    split; [ exact (env_rel_cw_bind cps wps vs cvars wvars hv hv' Hvars HW Hb
                      Hb') | ].
    split; [ exact Hst | repeat split ]. }
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds md));
    [ | exact I ].
  destruct (values_of_subkinds S vs (hm_params_subkinds md)) eqn:Hsk;
    [ | exact I ].
  rewrite (values_of_subkinds_same_values S' S vs _ (proj1 Hst)), Hsk.
  destruct (bind_params ps vs vars) as [hv | ] eqn:Hb; [ | exact I ].
  destruct (bind_params_rel ps vs vars vars' hv Hps Hvars Hb)
    as (hv' & Hb' & Hrel).
  rewrite Hb'. eexists. split; [ reflexivity | ].
  left. exists (without (map fst rec) D).
  split; [ exact Hh | split; [ exact Hrel | ] ].
  split; [ exact (rec_closures_rel D rec rec' vars vars' conts conts' Hrec
                    Hvars Hconts) | ].
  split; [ exact Hst | repeat split ].
Qed.

(* The call of the same closure in both runs, of code whose [my_closure]
   is plain, which is not an over-application: both runs enter related
   code, or give the same partial application to related frames. *)
Lemma ind_closure_answered (b : bool) (f : variable) (cl : value)
    (kind : call_kind) (vs : list value) (ra rta : arity)
    (r r' x x' : cont_value) (S S' : static_env) (rcd : read_record) :
  (∀ (id : code_id) (sl : list (value_slot * value)),
     closure_of S cl = Some (id, sl) -> RCH b f id) ->
  (∀ (id : code_id) (ci : code_info),
     RCH b f id -> CS id = Some ci -> ci_plain ci) ->
  (∀ (cl0 : value) (c0 : code) (ws rest : list value) (aa ra1 : arity),
     shape_of_closure_call S cl kind vs ra rta
     <> Some (Over_call cl0 c0 ws rest aa ra1)) ->
  cont_rel M CH CS RCH W REM r r' -> cont_rel M CH CS RCH W REM x x' ->
  statics_rel M CH CS RCH W REM S S' ->
  PGP.shape_over_ok (fun _ => True) S (PGP.cont_rem_ok REM S r)
    (shape_of_closure_call S cl kind vs ra rta) ->
  match
    run_call (fun p => apply_cont_value r [p] S empty_heap rcd)
      (shape_of_closure_call S cl kind vs ra rta) rta r x S empty_heap rcd
  with
  | Undefined => True
  | Exit k ws hp rd1 =>
    run_call (fun p => apply_cont_value r' [p] S' empty_heap rcd)
      (shape_of_closure_call S' cl kind vs ra rta) rta r' x' S' empty_heap
      rcd
    = Exit k ws hp rd1
  | Step s1 =>
    ∃ t1 : state,
      run_call (fun p => apply_cont_value r' [p] S' empty_heap rcd)
        (shape_of_closure_call S' cl kind vs ra rta) rta r' x' S'
        empty_heap rcd
      = Step t1
      ∧ related M CH CS RCH W REM s1 t1
  end.
Proof.
  intros Hrch Hmy Hno Hr Hx Hst Hsho.
  apply closure_call_answered; try assumption.
  - intros id sl ci Hcl Hci. exact (Hmy id ci (Hrch id sl Hcl) Hci).
  - intros cl0 c0 ws rest aa ra1 E. exfalso. exact (Hno _ _ _ _ _ _ E).
  - intros p E. rewrite E in Hsho.
    exact (cont_answered r r' [p] S S' rcd Hr Hst Hsho).
Qed.


(* A call of [rr_call_ind]: the same in both runs. *)
Lemma ind_call_step (D : list continuation) (ap : apply_expr) (f : variable)
    (vars vars' : env) (conts conts' : cont_env) (S S' : static_env)
    (rcd : read_record) :
  ap_callee ap = Var f -> plain_variable M f = true ->
  ind_targets CS RCH ap f ->
  forallb (plain_simple M) (ap_args ap) = true ->
  forallb (plain_simple M) (exn_extra_simples ap) = true ->
  ap_continuation ap ∉ D -> ap_exn_continuation ap ∉ D ->
  env_rel_c M CH vars vars' -> conts_rel M CH CS RCH W REM D conts conts' ->
  statics_rel M CH CS RCH W REM S S' ->
  flow_ok (State (Apply ap) vars conts S empty_heap rcd) ->
  PGP.step_rem_ok REM (State (Apply ap) vars conts S empty_heap rcd) ->
  match step (State (Apply ap) vars conts S empty_heap rcd) with
  | Undefined => True
  | Exit k ws hp rd1 =>
    step (State (Apply ap) vars' conts' S' empty_heap rcd)
    = Exit k ws hp rd1
  | Step s1 =>
    ∃ t1 : state,
      step (State (Apply ap) vars' conts' S' empty_heap rcd) = Step t1
      ∧ related M CH CS RCH W REM s1 t1
  end.
Proof.
  intros Hcallee Hf Hind Hargs Hextra Hk Hke Hvars Hconts Hst Hflow Hrem.
  assert (Hfv : eval_simple vars (Var f) = eval_simple vars' (Var f))
    by exact (plain_simple_eval M vars vars' (Var f) (proj1 Hvars) Hf).
  rewrite !step_apply, Hcallee, <- Hfv.
  rewrite <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hargs).
  rewrite <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hextra).
  cbn [eval_simple]. cbn [eval_simple] in Hfv.
  destruct (lookup f vars) as [v | ] eqn:Hv; [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs0;
    [ | exact I ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | exact I ].
  assert (Hsho : PGP.shape_over_ok (fun _ => True) S
                   (PGP.cont_rem_ok REM S
                      (resolve (ap_continuation ap) conts))
                   (shape_of_call S v (ap_call_kind ap) vs
                      (ap_args_arity ap) (ap_return_arity ap))).
  { unfold PGP.step_rem_ok in Hrem.
    cbn [state_expr state_vars state_statics state_conts] in Hrem.
    apply Hrem; [ rewrite Hcallee; exact Hv | exact Hvs0 ]. }
  destruct (Hflow ap f v eq_refl Hcallee Hv) as [Hpa Hrch].
  cbn [state_statics] in Hpa, Hrch.
  pose proof (resolve_rel D conts conts' _ Hconts Hk) as Hr.
  pose proof (cont_rel_with_extra _ _ extra
                (resolve_rel D conts conts' _ Hconts Hke))
    as Hx.
  assert (Htup : ∀ (id : code_id) (c : code),
            lookup id (static_code S) = Some c -> code_is_tupled c = false).
  { intros id c Hc.
    destruct (proj2 (proj2 Hst) id c Hc) as (c' & _ & ci & Hframe & _).
    exact (proj1 (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 Hframe))))))). }
  unfold call_function, ind_targets in *. revert Hsho.
  destruct (ap_call_kind ap) as [id | | ids] eqn:Hkind;
    [ contradiction | | ].
  - (* Of unknown arity, with one group of arguments. *)
    destruct Hind as [Hone Hmy].
    destruct (ap_args_arity ap) as [ | g [ | g' rest]] eqn:Haa;
      try discriminate Hone.
    assert (Hnp : ∀ (called : code_id) (sl : list (value_slot * value))
                    (c : code),
              closure_of S (call_closure v) = Some (called, sl) ->
              lookup called (static_code S) = Some c ->
              code_is_tupled c = false ∧ code_params_arity c <> []).
    { intros called sl c Hcl Hc.
      destruct (proj2 (proj2 Hst) called c Hc)
        as (c' & _ & ci & Hframe & _).
      pose proof Hframe as (Hci & _ & _ & _ & _ & _ & Htc & _ & Hlen
                            & _).
      split; [ exact Htc | ].
      destruct (Hmy called ci (proj2 (Hrch called sl) Hcl) Hci) as [_ Hne].
      intros He. rewrite He in Hlen. apply Hne. rewrite <- Hlen.
      reflexivity. }
    unfold shape_of_call.
    destruct (partial_application_of v) as [[[cl applied] aa0] | ] eqn:Hpv.
    + apply partial_application_of_some in Hpv. subst v.
      cbn beta iota.
      destruct (kinds_eqb (map kind_of_value vs) (unarize [g]));
        [ | intros _; exact I ].
      intros Hsho.
      apply (ind_closure_answered false f cl); try assumption.
      * intros id sl Hcl. exact (proj2 (Hrch id sl) Hcl).
      * intros id ci H1 H2. exact (proj1 (Hmy id ci H1 H2)).
      * intros cl0 c0 ws rest aa1 ra1.
        exact (partial_one_no_over_b S cl applied aa0 vs g
                 (ap_return_arity ap) cl0 c0 ws rest aa1 ra1
                 (Hpa cl applied aa0 eq_refl) Htup).
    + assert (Hcc : call_closure v = v)
        by (unfold call_closure; rewrite Hpv; reflexivity).
      rewrite Hcc in Hnp. intros Hsho.
      apply (ind_closure_answered false f v); try assumption.
      * intros id sl Hcl. apply (proj2 (Hrch id sl)). rewrite Hcc.
        exact Hcl.
      * intros id ci H1 H2. exact (proj1 (Hmy id ci H1 H2)).
      * intros cl0 c0 ws rest aa1 ra1.
        exact (CSP.closure_one_no_over S v Indirect_unknown_arity vs g
                 (ap_return_arity ap) cl0 c0 ws rest aa1 ra1 Hnp).
  - (* Of known arity. *)
    unfold shape_of_call. cbn beta iota. intros Hsho.
    apply (ind_closure_answered true f v); try assumption.
    + intros id sl Hcl. exact (proj1 (Hrch id sl) Hcl).
    + exact (CSP.known_arity_no_over S v ids vs (ap_args_arity ap)
               (ap_return_arity ap)).
Qed.
(* A call of [rr_call_over]: the same in both runs, which enter related
   code, through what is left of an over-application if it over-applies,
   or give the same partial application to related frames. *)
Lemma over_call_step (D : list continuation) (ap : apply_expr)
    (f : variable) (vars vars' : env) (conts conts' : cont_env)
    (S S' : static_env) (rcd : read_record) :
  ap_callee ap = Var f -> plain_variable M f = true ->
  ap_call_kind ap = Indirect_unknown_arity ->
  (∀ (id : code_id) (ci : code_info),
     RCH false f id ∨ REM id -> CS id = Some ci -> ci_plain ci)
  -> forallb (plain_simple M) (ap_args ap) = true ->
  forallb (plain_simple M) (exn_extra_simples ap) = true ->
  ap_continuation ap ∉ D -> ap_exn_continuation ap ∉ D ->
  env_rel_c M CH vars vars' -> conts_rel M CH CS RCH W REM D conts conts' ->
  statics_rel M CH CS RCH W REM S S' ->
  flow_ok (State (Apply ap) vars conts S empty_heap rcd) ->
  PGP.step_rem_ok REM (State (Apply ap) vars conts S empty_heap rcd) ->
  match step (State (Apply ap) vars conts S empty_heap rcd) with
  | Undefined => True
  | Exit k ws hp rd1 =>
    step (State (Apply ap) vars' conts' S' empty_heap rcd)
    = Exit k ws hp rd1
  | Step s1 =>
    ∃ t1 : state,
      step (State (Apply ap) vars' conts' S' empty_heap rcd) = Step t1
      ∧ related M CH CS RCH W REM s1 t1
  end.
Proof.
  intros Hcallee Hf Hkind Hmy Hargs Hextra Hk Hke Hvars Hconts Hst Hflow
    Hrem.
  assert (Hfv : eval_simple vars (Var f) = eval_simple vars' (Var f))
    by exact (plain_simple_eval M vars vars' (Var f) (proj1 Hvars) Hf).
  rewrite !step_apply, Hcallee, <- Hfv.
  rewrite <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hargs).
  rewrite <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hextra).
  cbn [eval_simple]. cbn [eval_simple] in Hfv.
  destruct (lookup f vars) as [v | ] eqn:Hv; [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs0;
    [ | exact I ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | exact I ].
  assert (Hsho : PGP.shape_over_ok (fun _ => True) S
                   (PGP.cont_rem_ok REM S
                      (resolve (ap_continuation ap) conts))
                   (shape_of_call S v (ap_call_kind ap) vs
                      (ap_args_arity ap) (ap_return_arity ap))).
  { unfold PGP.step_rem_ok in Hrem.
    cbn [state_expr state_vars state_statics state_conts] in Hrem.
    apply Hrem; [ rewrite Hcallee; exact Hv | exact Hvs0 ]. }
  destruct (Hflow ap f v eq_refl Hcallee Hv) as [_ Hrch].
  cbn [state_statics] in Hrch.
  pose proof (resolve_rel D conts conts' _ Hconts Hk) as Hr.
  pose proof (cont_rel_with_extra _ _ extra
                (resolve_rel D conts conts' _ Hconts Hke))
    as Hx.
  unfold call_function. rewrite Hkind in Hsho |- *. revert Hsho.
  unfold shape_of_call.
  destruct (partial_application_of v) as [[[cl applied] aa] | ] eqn:Hpv.
  - destruct (ap_args_arity ap) as [ | g ra']; [ intros _; exact I | ].
    destruct (kinds_eqb (map kind_of_value vs) (unarize (g :: ra')));
      [ | intros _; exact I ].
    intros Hsho.
    apply closure_call_answered; try assumption.
    + intros id sl ci Hcl Hci. apply (Hmy id ci); [ left | exact Hci ].
      apply (proj2 (Hrch id sl)). unfold call_closure. rewrite Hpv.
      exact Hcl.
    + intros _ _ _ _ _ _ _ id ci H1 H2. exact (Hmy id ci (or_intror H1) H2).
    + intros p Hp. rewrite Hp in Hsho.
      exact (cont_answered _ _ [p] S S' rcd Hr Hst Hsho).
  - intros Hsho.
    apply closure_call_answered; try assumption.
    + intros id sl ci Hcl Hci. apply (Hmy id ci); [ left | exact Hci ].
      apply (proj2 (Hrch id sl)). unfold call_closure. rewrite Hpv.
      exact Hcl.
    + intros _ _ _ _ _ _ _ id ci H1 H2. exact (Hmy id ci (or_intror H1) H2).
    + intros p Hp. rewrite Hp in Hsho.
      exact (cont_answered _ _ [p] S S' rcd Hr Hst Hsho).
Qed.


(* A call of known arity is a full call. *)
Lemma known_arity_full_shape (S : static_env) (v : value)
    (ids : option (list code_id)) (vs : list value) (ra rta : arity)
    (sh : call_shape) :
  shape_of_closure_call S v (Indirect_known_arity ids) vs ra rta = Some sh ->
  ∃ (cl0 : value) (c : code) (vs0 : list value) (aa : arity),
    sh = Full_call cl0 c vs0 aa.
Proof.
  unfold shape_of_closure_call. cbv zeta.
  destruct (closure_of S v) as [[called slots] | ]; [ | discriminate ].
  unfold known_arity_full.
  destruct ra as [ | a ra']; [ rewrite andb_false_r; discriminate | ].
  destruct (lookup called (static_code S)) as [c | ] eqn:Hc;
    [ | rewrite andb_false_r; discriminate ].
  destruct (code_ids_allow ids called); [ | discriminate ].
  destruct (negb (code_is_tupled c)) eqn:Ht; [ | discriminate ].
  destruct (Nat.eqb (length (a :: ra')) (length (code_params_arity c)))
    eqn:Hl; [ | discriminate ].
  cbn [andb]. unfold arguments_for_code.
  apply negb_true_iff in Ht. rewrite Ht. cbn beta iota.
  unfold shape_of_code_call. rewrite Hl. intros H. injection H as <-.
  eexists _, _, _, _. reflexivity.
Qed.

(* A call of a closure of code that is not tupled, with as many groups of
   arguments as the code has parameters, is a full call. *)
Lemma closure_call_full (S : static_env) (cl : value) (kind : call_kind)
    (vs : list value) (ra rta : arity) (sh : call_shape) :
  shape_of_closure_call S cl kind vs ra rta = Some sh ->
  (∀ (called : code_id) (slots : list (value_slot * value)) (c : code),
     closure_of S cl = Some (called, slots) ->
     lookup called (static_code S) = Some c ->
     code_is_tupled c = false ∧ length ra = length (code_params_arity c)) ->
  ∃ (cl0 : value) (c : code) (vs0 : list value) (aa : arity),
    sh = Full_call cl0 c vs0 aa.
Proof.
  intros H Hcodes.
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ H)
    as (called & slots & c & vs' & aa' & Hcl & Hc & _ & Hargs & Hcall).
  destruct (Hcodes called slots c Hcl Hc) as [Htup Hlen].
  unfold arguments_for_code in Hargs. rewrite Htup in Hargs.
  injection Hargs as <- <-.
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[_ ->] | [(Hlt & _) | (Hlt & _)]];
    [ eexists _, _, _, _; reflexivity | lia | lia ].
Qed.

(* The full call of the same closure in both runs, of code that never
   returns and whose [my_closure] is plain: both runs enter related
   code. *)
Lemma never_closure_entered (b : bool) (f : variable) (cl : value)
    (kind : call_kind) (vs : list value) (ra rta : arity)
    (r r' x x' : cont_value) (S S' : static_env) (rcd : read_record) :
  (∀ (id : code_id) (sl : list (value_slot * value)),
     closure_of S cl = Some (id, sl) -> RCH b f id) ->
  (∀ (id : code_id) (ci : code_info),
     RCH b f id -> CS id = Some ci ->
     ci_plain ci ∧ ci_never ci = true) ->
  (∀ sh : call_shape,
     shape_of_closure_call S cl kind vs ra rta = Some sh ->
     ∃ (cl0 : value) (c : code) (vs0 : list value) (aa : arity),
       sh = Full_call cl0 c vs0 aa) ->
  statics_rel M CH CS RCH W REM S S' ->
  match
    run_call (fun p => apply_cont_value r [p] S empty_heap rcd)
      (shape_of_closure_call S cl kind vs ra rta) rta r x S empty_heap rcd
  with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ (cc cc' : code) (hv hv' : env) (ci : code_info),
      s1 = State (code_body cc) hv
             [(code_return_continuation cc, r);
              (code_exn_continuation cc, x)]
             S empty_heap rcd
      ∧ run_call (fun p => apply_cont_value r' [p] S' empty_heap rcd)
          (shape_of_closure_call S' cl kind vs ra rta) rta r' x' S'
          empty_heap rcd
        = Step (State (code_body cc') hv'
                  [(code_return_continuation cc, r');
                   (code_exn_continuation cc, x')]
                  S' empty_heap rcd)
      ∧ env_rel_c M CH hv hv'
      ∧ rr M CH CS RCH W REM (code_dead ci cc) (code_body cc) (code_body cc')
      ∧ ci_never ci = true
  end.
Proof.
  intros Hrch Hmy Hfull Hst.
  pose proof (closure_shape_rel S S' cl kind vs ra rta
                (fun id sl ci H1 H2 =>
                   proj2 (proj1 (Hmy id ci (Hrch id sl H1) H2))) Hst)
    as Hsh.
  destruct (shape_of_closure_call S cl kind vs ra rta) as [sh | ] eqn:E;
    [ | exact I ].
  destruct (Hfull sh eq_refl) as (cl0 & c & vs0 & aa & ->).
  destruct Hsh as (called & sl & c' & ci & Hcl & -> & Hframe & Hbody & Hsh).
  rewrite Hsh. cbn [run_call].
  destruct (Hmy called ci (Hrch called sl Hcl) (proj1 Hframe))
    as [[Hnone Hpl] Hnv].
  assert (Hclass : class_rel (ci_my_closure ci) cl cl)
    by (rewrite Hnone; reflexivity).
  pose proof (call_code_entered called ci c c' cl cl vs0 aa rta r r' x x'
                S S' rcd Hframe Hbody Hclass (proj1 Hst) Hpl)
    as Hentered.
  destruct (call_code cl c vs0 aa rta r x S empty_heap rcd) as [s1 | | ];
    [ | contradiction | exact I ].
  destruct Hentered as (hv & hv' & -> & Hcall & Hrel).
  exists c, c', hv, hv', ci.
  split; [ reflexivity | split; [ exact Hcall | ] ].
  split; [ exact Hrel | split; [ exact Hbody | exact Hnv ] ].
Qed.

(* A call of code that never returns ([never_ok]), the second run
   returning to [k2]: both runs enter related code, the frames of their
   return continuations unrelated. *)
Lemma never_entered (ap : apply_expr) (k2 : continuation) (vars vars' : env)
    (conts conts' : cont_env) (S S' : static_env) (rcd : read_record) :
  never_ok M CH CS RCH ap ->
  env_rel_c M CH vars vars' -> statics_rel M CH CS RCH W REM S S' ->
  flow_ok (State (Apply ap) vars conts S empty_heap rcd) ->
  match step (State (Apply ap) vars conts S empty_heap rcd) with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ (cc cc' : code) (hv hv' : env) (extra : list value) (ci : code_info),
      s1 = State (code_body cc) hv
             [(code_return_continuation cc,
               resolve (ap_continuation ap) conts);
              (code_exn_continuation cc,
               with_extra_args (resolve (ap_exn_continuation ap) conts)
                 extra)]
             S empty_heap rcd
      ∧ step (State (Apply (with_continuation ap k2)) vars' conts' S'
                empty_heap rcd)
        = Step (State (code_body cc') hv'
                  [(code_return_continuation cc, resolve k2 conts');
                   (code_exn_continuation cc,
                    with_extra_args (resolve (ap_exn_continuation ap) conts')
                      extra)]
                  S' empty_heap rcd)
      ∧ env_rel_c M CH hv hv'
      ∧ rr M CH CS RCH W REM (code_dead ci cc) (code_body cc) (code_body cc')
      ∧ ci_never ci = true
  end.
Proof.
  intros [(id & ci & Hok & Hnv) | (f & Hcallee & Hf & Hargs & Hextra & Hind)]
    Hvars Hst Hflow.
  - pose proof (call_entered ap k2 id ci vars vars' conts conts' S S' rcd Hok
                  Hvars Hst)
      as Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | | ]; [ | exact Hc | exact I ].
    destruct Hc as (cc & cc' & hv & hv' & extra & Hs1 & Ht & Hrel & Hbody).
    exists cc, cc', hv, hv', extra, ci.
    split; [ exact Hs1 | split; [ exact Ht | ] ].
    split; [ exact Hrel | split; [ exact Hbody | exact Hnv ] ].
  - assert (Hfv : eval_simple vars (Var f) = eval_simple vars' (Var f))
      by exact (plain_simple_eval M vars vars' (Var f) (proj1 Hvars) Hf).
    rewrite !step_apply.
    change (exn_extra_simples (with_continuation ap k2))
      with (exn_extra_simples ap).
    cbn [with_continuation ap_callee ap_call_kind ap_args ap_continuation
         ap_exn_continuation ap_args_arity ap_return_arity].
    rewrite Hcallee, <- Hfv.
    rewrite <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hargs).
    rewrite <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hextra).
    cbn [eval_simple].
    destruct (lookup f vars) as [v | ] eqn:Hv; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    destruct (Hflow ap f v eq_refl Hcallee Hv) as [Hpa Hrch].
    cbn [state_statics] in Hpa, Hrch.
    set (r := resolve (ap_continuation ap) conts).
    set (x := with_extra_args (resolve (ap_exn_continuation ap) conts)
                extra).
    set (r' := resolve k2 conts').
    set (x' := with_extra_args (resolve (ap_exn_continuation ap) conts')
                 extra).
    unfold call_function, ind_never in *.
    assert (Hgen :
      ∀ (b : bool) (cl : value) (kind : call_kind) (vs0 : list value)
        (ra : arity),
        (∀ (id : code_id) (sl : list (value_slot * value)),
           closure_of S cl = Some (id, sl) -> RCH b f id) ->
        (∀ (id : code_id) (ci : code_info),
           RCH b f id -> CS id = Some ci ->
           ci_plain ci ∧ ci_never ci = true) ->
        (∀ sh : call_shape,
           shape_of_closure_call S cl kind vs0 ra (ap_return_arity ap)
           = Some sh ->
           ∃ (cl0 : value) (c : code) (vs1 : list value) (aa : arity),
             sh = Full_call cl0 c vs1 aa) ->
        match
          run_call (fun p => apply_cont_value r [p] S empty_heap rcd)
            (shape_of_closure_call S cl kind vs0 ra (ap_return_arity ap))
            (ap_return_arity ap) r x S empty_heap rcd
        with
        | Undefined => True
        | Exit _ _ _ _ => False
        | Step s1 =>
          ∃ (cc cc' : code) (hv hv' : env) (extra0 : list value)
            (ci : code_info),
            s1 = State (code_body cc) hv
                   [(code_return_continuation cc, r);
                    (code_exn_continuation cc,
                     with_extra_args (resolve (ap_exn_continuation ap) conts)
                       extra0)]
                   S empty_heap rcd
            ∧ run_call (fun p => apply_cont_value r' [p] S' empty_heap rcd)
                (shape_of_closure_call S' cl kind vs0 ra
                   (ap_return_arity ap))
                (ap_return_arity ap) r' x' S' empty_heap rcd
              = Step (State (code_body cc') hv'
                        [(code_return_continuation cc, r');
                         (code_exn_continuation cc,
                          with_extra_args
                            (resolve (ap_exn_continuation ap) conts')
                            extra0)]
                        S' empty_heap rcd)
            ∧ env_rel_c M CH hv hv'
            ∧ rr M CH CS RCH W REM (code_dead ci cc) (code_body cc)
                (code_body cc')
            ∧ ci_never ci = true
        end).
    { intros b cl kind vs0 ra Hr Hm Hfull.
      pose proof (never_closure_entered b f cl kind vs0 ra
                    (ap_return_arity ap) r r' x x' S S' rcd Hr Hm Hfull Hst)
        as H.
      destruct (run_call (fun p => apply_cont_value r [p] S empty_heap rcd)
                  (shape_of_closure_call S cl kind vs0 ra
                     (ap_return_arity ap))
                  (ap_return_arity ap) r x S empty_heap rcd)
        as [s1 | | ]; [ | exact H | exact I ].
      destruct H as (cc & cc' & hv & hv' & ci & Hs1 & Ht & Hrel & Hbody
                     & Hnv).
      exists cc, cc', hv, hv', extra, ci.
      split; [ exact Hs1 | split; [ exact Ht | ] ].
      split; [ exact Hrel | split; [ exact Hbody | exact Hnv ] ]. }
    destruct (ap_call_kind ap) as [id | | ids] eqn:Hkind;
      [ contradiction | | ].
    + (* Of unknown arity, with one group of arguments, of code with one
         parameter. *)
      destruct Hind as [Hone Hmy].
      destruct (ap_args_arity ap) as [ | g [ | g' rest]] eqn:Haa;
        try discriminate Hone.
      assert (Hone_param :
        ∀ (cl : value) (called : code_id) (sl : list (value_slot * value))
          (c : code),
          closure_of S (call_closure v) = Some (called, sl) ->
          lookup called (static_code S) = Some c ->
          code_is_tupled c = false ∧ length (code_params_arity c) = 1).
      { intros cl called sl c Hcl Hc.
        destruct (proj2 (proj2 Hst) called c Hc)
          as (c' & _ & ci & Hframe & _).
        pose proof Hframe as (Hci & _ & _ & _ & _ & _ & Htc & _ & Hlen
                              & _).
        split; [ exact Htc | ].
        destruct (Hmy called ci (proj2 (Hrch called sl) Hcl) Hci)
          as (_ & _ & H1). rewrite Hlen. exact H1. }
      unfold shape_of_call.
      destruct (partial_application_of v) as [[[cl applied] aa0] | ]
        eqn:Hpv.
      * apply partial_application_of_some in Hpv. subst v.
        cbn beta iota.
        destruct (kinds_eqb (map kind_of_value vs) (unarize [g]));
          [ | exact I ].
        apply (Hgen false cl).
        -- intros id sl Hcl. exact (proj2 (Hrch id sl) Hcl).
        -- intros id ci H1 H2. destruct (Hmy id ci H1 H2) as (Hn & Hv' & _).
           exact (conj Hn Hv').
        -- intros sh Hsh. apply (closure_call_full _ _ _ _ _ _ sh Hsh).
           intros called sl c Hcl Hc.
           destruct (Hone_param cl called sl c Hcl Hc) as [Htc Hlen].
           split; [ exact Htc | ].
           destruct (Hpa cl applied aa0 eq_refl)
             as (called' & sl' & c' & Hcl' & Hc' & Hlt).
           cbn [call_closure partial_application_of] in Hcl.
           rewrite Hcl in Hcl'. injection Hcl' as <- _.
           rewrite Hc in Hc'. injection Hc' as <-.
           rewrite Hlen in Hlt |- *. rewrite length_app. cbn [length].
           destruct aa0; [ reflexivity | cbn [length] in Hlt; lia ].
      * assert (Hcc : call_closure v = v)
          by (unfold call_closure; rewrite Hpv; reflexivity).
        apply (Hgen false v).
        -- intros id sl Hcl. apply (proj2 (Hrch id sl)). rewrite Hcc.
           exact Hcl.
        -- intros id ci H1 H2. destruct (Hmy id ci H1 H2) as (Hn & Hv' & _).
           exact (conj Hn Hv').
        -- intros sh Hsh. apply (closure_call_full _ _ _ _ _ _ sh Hsh).
           intros called sl c Hcl Hc. rewrite <- Hcc in Hcl.
           destruct (Hone_param v called sl c Hcl Hc) as [Htc Hlen].
           split; [ exact Htc | rewrite Hlen; reflexivity ].
    + (* Of known arity. *)
      unfold shape_of_call. cbn beta iota.
      apply (Hgen true v).
      * intros id sl Hcl. exact (proj1 (Hrch id sl) Hcl).
      * exact Hind.
      * intros sh Hsh. exact (known_arity_full_shape S v ids vs _ _ sh Hsh).
Qed.

Lemma main_answered (s t : state) :
  related_main M CH CS RCH W REM s t -> flow_ok s -> PGP.step_rem_ok REM s ->
  answered (related M CH CS RCH W REM) eq measure s t.
Proof.
  destruct s as [p vars conts S hp rcd],
    t as [q vars' conts' S' hp' rcd'].
  intros (D & Hrr & Hvars & Hconts & Hst & Hhp & Hhp' & Hrcd) Hflow Hrem.
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst hp hp' rcd'. unfold answered.
  destruct Hrr as
    [D0 x kind d body body' Hx Hd Hnh Hbody
    | D0 x kind slot decls vs body body' Hx Hvs Hbody
    | D0 x d fields ls body body' Hfields Hxc Hxf Hl Hoff Hbody
    | D0 x y fields fields_y pairs body body' Hfields Hxc Hxf Hy Hne Hl
        Hbody
    | D0 x d f s y fields fields_y nested pairs body body' Hfields Hxc Hxf
        Hread Hs Hy Hne Hin Hl Hbody
    | D0 x kind d f s y v fields body body' Hx Hread Hs Hfields Hin Hbody
    | D0 x r decls vs body body' Hxr Hdecls Hok Hnd Hbody
    | D0 x kind d w s y r sl body body' Hx Hd Hs Hyr Hin Hbody
    | D0 x d w s y r fields nested pairs body body' Hfields Hxc Hxf Hd Hs
        Hyr Hin Hl Hbody
    | D0 bs bs' body body' Hbs Hbody
    | D0 k ps md h h' body body' Hps Hh Hbody
    | D0 hs hs' body body' Hhs Hbody
    | D0 ac Hac Hk
    | D0 scrutinee arms Hscrutinee Harms Hks
    | D0
    | D0 ap id ci args' aa' Hkind Hci Hcallee Hargs Hextra Hlen Hk Hke
    | D0 ap f Hcallee Hf Hind Hargs Hextra Hk Hke
    | D0 ap k Hok Hk Hke
    | D0 ap w wps wmd Hok Hw Hke
    | D0 ap wc cps cmd hb w wps wmd Hok Hwc Hw Hke
    | D0 k h body body' Hnk Hbody
    | D0 ap id ci k wc cps cmd cargs w wps wmd wargs Hok Hwc Hw Hk Hke HW
        Hwrap
    | D0 ap f Hcallee Hf Hkind Hmy Hargs Hextra Hk Hke ].
  - (* A [Let] of both. *)
    rewrite step_let_exec,
      (named_plain_exec_c S S' empty_heap rcd vars vars' d (proj1 Hst)
         (proj1 Hvars) Hd).
    destruct (exec_named S' empty_heap rcd vars' d)
      as [[[w hp1] rcd1] | ] eqn:Hw; [ | exact I ].
    destruct (exec_named_empty_heap S' rcd vars' d w hp1 rcd1 Hnh Hw)
      as (_ & -> & ->).
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hk; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_exec, Hw, Hk; reflexivity | ].
    left; exists D0; split; [ exact Hbody | ].
    split; [ exact (env_rel_c_bind _ _ x w Hvars Hx) | ].
    split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* A plain set of closures. *)
    rewrite step_let_exec. cbn [exec_named eval_named].
    rewrite (plain_slots_eval vars vars' vs (proj1 Hvars) Hvs).
    destruct (lookup slot decls) as [code | ] eqn:Hcode; [ | exact I ].
    destruct (eval_value_slots vars' vs) as [vs1 | ] eqn:Hvs1; [ | exact I ].
    cbn. destruct (kind_eqb Value kind) eqn:Hk; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_exec; cbn [exec_named eval_named];
             rewrite Hcode, Hvs1; cbn; rewrite Hk; reflexivity | ].
    left; exists D0; split; [ exact Hbody | ].
    split; [ exact (env_rel_c_bind _ _ x _ Hvars Hx) | ].
    split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* The allocation of an unboxed variable. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[w hp1] rcd1] | ]
      eqn:Hxw; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d w hp1 rcd1 Hoff Hxw)
      as (Hw & -> & ->).
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (lets_case_c vars vars' x fields w (concat ls) body' conts' S'
                empty_heap rcd Hvars Hfields Hxc)
      as (vars1 & Hrun & Hrel).
    + clear -Hl. induction Hl as [ | [f u] l fields ls Hfu _ IH];
        [ reflexivity | ].
      cbn [concat]. rewrite map_app, IH. unfold leaf_variables.
      rewrite leaves_cons, map_app. f_equal.
      destruct u as [v | nested]; cbn [alloc_entry] in Hfu.
      * destruct Hfu as (kind & s & -> & _). reflexivity.
      * destruct Hfu as (b & fb & pairs & _ & _ & _ & Hpairs & ->).
        rewrite pairs_lets_variables.
        destruct (pairs_spec _ _ _ _ Hpairs) as (_ & _ & Hc & _).
        rewrite Hc. reflexivity.
    + intros v kind s v' Hin Hv' ->.
      destruct (forall2_concat_in _ _ _ _ Hl Hin)
        as ([f u] & l0 & _ & Hent & Hin0).
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * destruct Hent as (kind0 & s0 & -> & _ & Hplain).
        destruct Hin0 as [Heq | []]. injection Heq as <- <- ->.
        simpl in Hplain. apply plain_variable_iff in Hplain as [_ Hnot].
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
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * destruct Hent as (kind0 & s0 & -> & Hf & Hplain).
        destruct Hin0 as [Heq | []]. injection Heq as <- <- <-.
        destruct (alloc_field_eval S vars d w f kind0 s0 Hw Hf)
          as (w' & Hs & _ & Hkw).
        exists w'. split; [ | exact Hkw ].
        rewrite <- (plain_simple_eval M vars vars' s0 (proj1 Hvars) Hplain).
        exact Hs.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval S vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & _ & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 (proj1 Hvars) b fb wb Hb Hs))
          as (_ & Hseconds & _).
        apply in_pairs_lets in Hin0 as (v' & -> & Hin').
        destruct (Hseconds v v' kind Hin') as [_ (w' & Hl' & Hk')].
        exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k Hpath.
      apply upath_unboxed_inv in Hpath as (f & u & p0 & -> & Hfu & Hp0).
      destruct (forall2_in_both _ _ _ _ Hl Hfu) as (l0 & Hl0 & Hent).
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * apply upath_not_unboxed_inv in Hp0 as (-> & -> & ->).
        destruct Hent as (kind0 & s0 & -> & Hf & Hplain).
        pose proof (alloc_field_kind d f kind0 s0 Hf) as Hk0. subst kind0.
        destruct (alloc_field_eval S vars d w f (field_kind f) s0 Hw Hf)
          as (w' & Hs & Hfw & _).
        exists s0, w'. split; [ exact (in_concat_of _ _ _ Hl0 (or_introl
                                                       eq_refl)) | ].
        split; [ rewrite <- (plain_simple_eval M vars vars' s0 (proj1 Hvars)
                               Hplain);
                 exact Hs | ].
        cbn [path_value]. rewrite Hfw. reflexivity.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval S vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & Hfw & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 (proj1 Hvars) b fb wb Hb Hs))
          as (_ & _ & Hpaths).
        destruct (Hpaths p0 v k (upath_unboxed_kind _ Value _ _ _ _ Hp0))
          as (v' & w' & Hin' & Hl' & Hpw).
        exists (Var v'), w'.
        split; [ exact (in_concat_of _ _ _ Hl0
                          (pairs_in_lets pairs v v' k Hin')) | ].
        split; [ exact Hl' | ]. cbn [path_value]. rewrite Hfw. exact Hpw.
    + apply (run_answer (related M CH CS RCH W REM) _ _ _ _ (length (concat ls))
               Hrun).
      * left; exists D0; split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* An alias of an unboxed variable. *)
    rewrite step_let_simple. cbn [eval_simple].
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (pairs_hold vars' fields fields_y w pairs Hl
                (proj2 (proj1 Hvars) y fields_y w Hy Hw))
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case_c vars vars' x fields w (pairs_lets pairs) body'
                conts' S' empty_heap rcd Hvars Hfields Hxc)
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
    + apply (run_answer (related M CH CS RCH W REM) _ _ _ _
               (length (pairs_lets pairs)) Hrun).
      * left; exists D0; split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* The load of a field unboxed in turn. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[r0 hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d r0 hp1 rcd1
                (read_not_heap d f s Hread) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r0) Value) eqn:Hk; [ | exact I ].
    destruct (read_eval S vars d f s r0 Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    pose proof (fields_hold_nested vars' fields_y nested f w0 r0
                  (proj2 (proj1 Hvars) y fields_y w0 Hy Hw0) Hin Hfield)
      as Hhold.
    destruct (pairs_hold vars' fields nested r0 pairs Hl Hhold)
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case_c vars vars' x fields r0 (pairs_lets pairs) body'
                conts' S' empty_heap rcd Hvars Hfields Hxc)
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
    + apply (run_answer (related M CH CS RCH W REM) _ _ _ _
               (length (pairs_lets pairs)) Hrun).
      * left; exists D0; split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* A read of a field. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[r0 hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d r0 hp1 rcd1
                (read_not_heap d f s Hread) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r0) kind) eqn:Hk; [ | exact I ].
    destruct (read_eval S vars d f s r0 Hread Hr)
      as (w0 & Hw0 & Hnc & Hfield).
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    assert (Hp : upath Value (Unboxed fields) [f] v (field_kind f))
      by (econstructor; [ exact Hin | constructor ]).
    destruct (proj2 (proj1 Hvars) y fields w0 Hfields Hw0 [f] v _ Hp)
      as (w' & Hpw & Hv & _).
    cbn [path_value] in Hpw.
    destruct (field_value f w0) as [w1 | ] eqn:Hfw; [ | discriminate Hpw ].
    injection Hpw as <-.
    rewrite <- (Hfield w1 eq_refl) in Hv.
    right. left. eexists. split.
    + rewrite step_let_simple. cbn [eval_simple]. rewrite Hv, Hk.
      reflexivity.
    + left; exists D0; split; [ exact Hbody | ].
      split; [ exact (env_rel_c_bind _ _ x r0 Hvars Hx) | ].
      split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* A set whose representation changes. *)
    rewrite step_let_exec. cbn [exec_named eval_named].
    destruct (lookup (cr_current r) decls) as [code | ] eqn:Hcode;
      [ | exact I ].
    destruct (eval_value_slots vars vs) as [vs1 | ] eqn:Hvs1; [ | exact I ].
    destruct (new_slots_eval vars vars' r vs vs1 (proj1 Hvars) Hok Hnd Hvs1)
      as (vs2 & Hvs2 & Hslots).
    cbn. right. left. eexists. split.
    + rewrite step_let_exec. cbn [exec_named eval_named].
      rewrite Hdecls, Hvs2. reflexivity.
    + left; exists D0; split; [ exact Hbody | ].
      split; [ | split; [ exact Hconts | split; [ exact Hst | repeat split ]]].
      apply (env_rel_c_bind_changed vars vars' x r _ _ Hvars Hxr).
      apply (set_crel_exact vars' r decls vs vs1 vs2 code Hcode);
        [ rewrite Hdecls, Hcode; reflexivity | exact Hok | exact Hnd
        | exact Hvs2 | exact Hslots ].
  - (* A read of a new slot. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[r0 hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d r0 hp1 rcd1
                (slot_read_not_heap d w s Hd) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r0) kind) eqn:Hk; [ | exact I ].
    destruct (slot_read_var S vars d w s r0 Hd Hr) as (v1 & Hv1).
    destruct Hs as [-> | (k & ->)];
      [ | rewrite (slot_read_const S vars d w _ Hd) in Hr; discriminate Hr ].
    cbn [eval_simple] in Hv1.
    destruct (proj2 Hvars y r v1 Hyr Hv1) as (v2 & Hv2 & Hc).
    destruct (slot_read_crel S S' vars d w (Var y) y r v1 v2 r0 Hd
                (or_introl eq_refl) Hv1 (proj1 Hc) Hr)
      as (vs' & Hholds & Hproj).
    pose proof (Hholds _ Hin) as Hsl. cbn [slot_holds] in Hsl.
    right. left. eexists. split.
    + rewrite step_let_exec, exec_named_proj,
        (eval_proj_from S' vars' _ sl y v2 Hv2), Hproj, Hsl, Hk.
      reflexivity.
    + left; exists D0; split; [ exact Hbody | ].
      split; [ exact (env_rel_c_bind _ _ x r0 Hvars Hx) | ].
      split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* The reads of the new slots of an unboxed value. *)
    rewrite step_let_exec.
    destruct (exec_named S empty_heap rcd vars d) as [[[r0 hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (exec_named_empty_heap S rcd vars d r0 hp1 rcd1
                (slot_read_not_heap d w s Hd) Hxr)
      as (Hr & -> & ->).
    destruct (kind_eqb (kind_of_value r0) Value) eqn:Hk; [ | exact I ].
    destruct (slot_read_var S vars d w s r0 Hd Hr) as (v1 & Hv1).
    destruct Hs as [-> | (k & ->)];
      [ | rewrite (slot_read_const S vars d w _ Hd) in Hr; discriminate Hr ].
    cbn [eval_simple] in Hv1.
    destruct (proj2 Hvars y r v1 Hyr Hv1) as (v2 & Hv2 & Hc).
    destruct (slot_read_crel S S' vars d w (Var y) y r v1 v2 r0 Hd
                (or_introl eq_refl) Hv1 (proj1 Hc) Hr)
      as (vs' & Hholds & Hproj).
    pose proof (Hholds _ Hin) as Hhold. cbn [slot_holds] in Hhold.
    destruct (pairs_hold vs' fields nested r0 pairs Hl Hhold)
      as (Hfirsts & Hseconds & Hpaths).
    destruct (proj_lets_case vars vars' x y fields r0 (current_slot r) pairs
                body' conts' S' empty_heap rcd Hvars Hfields Hxc Hfirsts)
      as (vars1 & Hrun & Hrel).
    + intros Hin'.
      pose proof (field_variables_in M x fields y Hfields Hin') as Hfv.
      apply among_In in Hfv. rewrite (CH_fresh y r Hyr) in Hfv.
      discriminate Hfv.
    + intros v s0 k Hin0.
      destruct (Hseconds v s0 k Hin0) as [_ (w' & Hl' & Hk')].
      exists w'. rewrite (eval_proj_from S' vars' _ s0 y v2 Hv2), Hproj.
      split; assumption.
    + intros p v k Hp.
      destruct (Hpaths p v k Hp) as (s0 & w' & Hin0 & Hl' & Hpw).
      exists s0, w'. split; [ exact Hin0 | ].
      rewrite (eval_proj_from S' vars' _ s0 y v2 Hv2), Hproj.
      split; assumption.
    + apply (run_answer (related M CH CS RCH W REM) _ _ _ _ (length pairs)
               Hrun).
      * left; exists D0; split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
      * intros _. unfold measure. simpl. lia.
  - (* A [Let_static]. *)
    rewrite step_let_static.
    destruct (eval_static_bindings vars bs S) as [S1 | ] eqn:Hev;
      [ | exact I ].
    destruct (bindings_eval_rel bs bs' Hbs vars vars' S S' S1 Hvars Hst Hev)
      as (S1' & Hev' & Hst1).
    right. left. eexists.
    split; [ rewrite step_let_static, Hev'; reflexivity | ].
    left; exists D0; split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ exact Hconts | split; [ exact Hst1 | repeat split ] ].
  - (* A non-recursive [Let_cont]. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    left. exists (without [k] D0).
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    intros k0 Hk0. cbn [state_conts]. unfold resolve. rewrite !lookup_cons.
    destruct (Pos.eqb k0 k) eqn:E.
    + apply (cont_rel_closure M CH CS RCH W REM D0 ps md h h' vars vars' conts
               conts' [] [] Hps); cbn [map]; rewrite ?without_nil;
        [ exact Hh | exact Hvars | exact Hconts | constructor ].
    + apply Hconts. intros H. apply Hk0. apply in_without_iff.
      split; [ exact H | ]. intros [He | []]. subst k0.
      rewrite Pos.eqb_refl in E. discriminate E.
  - (* A recursive [Let_cont]. *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics hs) eqn:Hb; [ exact I | ].
    assert (Hb' : handlers_bind_statics hs' = false).
    { destruct (handlers_bind_statics hs') eqn:E; [ | reflexivity ].
      rewrite (proj2 (proj2 (proj1 (proj2 rr_shape) _ hs hs' Hhs)) E) in Hb.
      discriminate Hb. }
    right. left. eexists. split.
    { rewrite step_let_cont_recursive, Hb'. reflexivity. }
    left. exists (without (map fst hs) D0).
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ apply rec_closures_rel; assumption | ].
    split; [ exact Hst | repeat split ].
  - (* An [Apply_cont]. *)
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    unfold PGP.step_rem_ok in Hrem.
    cbn [state_expr state_vars state_conts state_statics] in Hrem.
    pose proof (cont_answered _ _ vs S S' rcd
                  (resolve_rel D0 conts conts' (ac_continuation ac) Hconts
                     Hk)
                  Hst (Hrem vs Hvs))
      as Ha.
    destruct (apply_cont_value (resolve (ac_continuation ac) conts) vs S
                empty_heap rcd)
      as [s1 | k ws hp1 rd1 | ] eqn:Happ; [ | | exact I ].
    + destruct Ha as (t1 & Ht1 & Hrel). right. left. exists t1.
      split; [ | exact Hrel ].
      rewrite step_apply_cont,
        <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hac), Hvs.
      unfold apply_continuation. exact Ht1.
    + exists ws, rd1. split; [ | reflexivity ].
      rewrite step_apply_cont,
        <- (plain_simples_eval M vars vars' _ (proj1 Hvars) Hac), Hvs.
      unfold apply_continuation. exact Ha.
  - (* A [Switch]. *)
    rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    right. left. eexists. split.
    + rewrite step_switch,
        <- (plain_simple_eval M vars vars' scrutinee (proj1 Hvars)
              Hscrutinee),
        Hv, Hi, Harm.
      reflexivity.
    + left. exists D0. split; [ | split; [ exact Hvars | ] ].
      * destruct (find_arm_in i arms ac Harm) as (d & Hin).
        constructor; [ | exact (Hks _ Hin) ].
        rewrite forallb_forall in Harms. exact (Harms _ Hin).
      * split; [ exact Hconts | split; [ exact Hst | repeat split ] ].
  - (* [Invalid]. *)
    rewrite step_invalid. exact I.
  - (* A call. *)
    pose proof (call_entered_u ap (with_args ap args' aa') id ci vars vars'
                  conts conts' S S' rcd Hkind Hci Hcallee Hextra Hlen eq_refl
                  eq_refl eq_refl eq_refl eq_refl Hargs Hvars Hst)
      as Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | | ]; [ | contradiction | exact I ].
    destruct Hc as (cc & cc' & hv & hv' & extra & -> & Ht & Hrel & Hcbody).
    right. left. eexists. split; [ exact Ht | ].
    left. exists (code_dead ci cc).
    split; [ exact Hcbody | split; [ exact Hrel | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    apply code_conts_rel;
      [ intros _; apply Hconts; assumption
      | apply cont_rel_with_extra; apply Hconts; assumption ].
  - (* A call whose code is not known. *)
    pose proof (ind_call_step D0 ap f vars vars' conts conts' S S' rcd
                  Hcallee Hf Hind Hargs Hextra Hk Hke Hvars Hconts Hst Hflow
                  Hrem)
      as Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | k ws hp1 rd1 | ]; [ | | exact I ].
    + destruct Hc as (t1 & Ht1 & Hrel). right. left. exists t1.
      split; [ exact Ht1 | exact Hrel ].
    + exists ws, rd1. split; [ exact Hc | reflexivity ].
  - (* A call of code that never returns. *)
    pose proof (never_entered ap k vars vars' conts conts' S S' rcd Hok
                  Hvars Hst Hflow)
      as Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | | ]; [ | contradiction | exact I ].
    destruct Hc
      as (cc & cc' & hv & hv' & extra & ci & -> & Ht & Hrel & Hcbody & Hnv).
    right. left. eexists. split; [ exact Ht | ].
    left. exists (code_dead ci cc).
    split; [ exact Hcbody | split; [ exact Hrel | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    apply code_conts_rel;
      [ intros H; rewrite Hnv in H; discriminate H
      | apply cont_rel_with_extra, Hconts; exact Hke ].
  - (* The same, the second run returning to a handler [Invalid]. *)
    pose proof (never_entered ap w vars vars' conts
                  ((w, Closure (Handler wps wmd Invalid) vars' conts' [])
                   :: conts')
                  S S' rcd Hok Hvars Hst Hflow)
      as Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | | ]; [ | contradiction | exact I ].
    destruct Hc
      as (cc & cc' & hv & hv' & extra & ci & -> & Ht & Hrel & Hcbody & Hnv).
    right. right. right. eexists. exists 0. split.
    { exact (run_two _ _ _ (step_let_cont_non_recursive _ _ _ _ _ _ _ _) Ht). }
    left. exists (code_dead ci cc).
    split; [ exact Hcbody | split; [ exact Hrel | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    rewrite (resolve_other (ap_exn_continuation ap) w)
      by (intros He; exact (Hw (eq_sym He))).
    apply code_conts_rel;
      [ intros H; rewrite Hnv in H; discriminate H
      | apply cont_rel_with_extra, Hconts; exact Hke ].
  - (* The same, the first run returning to a wrapper. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    left. exists (wc :: w :: D0).
    split; [ | split; [ exact Hvars | ] ].
    { apply (rr_call_never M CH CS RCH W REM _ ap w Hok);
        [ right; left; reflexivity | ].
      intros [He | [He | He]];
        [ exact (Hwc He) | exact (Hw He) | exact (Hke He) ]. }
    split; [ | split; [ exact Hst | repeat split ] ].
    intros k0 Hk0. cbn [state_conts].
    rewrite (resolve_other k0 wc), (resolve_other k0 w);
      [ | intros ->; apply Hk0; right; left; reflexivity
        | intros ->; apply Hk0; left; reflexivity ].
    apply Hconts. intros H. apply Hk0. right. right. exact H.
  - (* A handler that only the first run binds. *)
    rewrite step_let_cont_non_recursive. right. right. left.
    split; [ unfold measure; simpl; lia | ].
    left. exists (k :: D0).
    split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | split; [ exact Hst | repeat split ] ].
    intros k0 Hk0. cbn [state_conts].
    rewrite (resolve_other k0 k) by (intros ->; apply Hk0; left; reflexivity).
    apply Hconts. intros H. apply Hk0. right. exact H.
  - (* A call that returns through a wrapper of each run: both bind their
       wrapper, and are at the call. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    right. left.
    exists (with_continuation ap wc), w, id, ci.
    cbn [state_expr state_vars state_conts state_statics state_heap
         state_reads].
    split; [ reflexivity | split; [ reflexivity | ] ].
    split; [ exact Hok | split; [ exact Hvars | ] ].
    split.
    { cbn [with_continuation ap_continuation]. rewrite !resolve_head.
      exact (cont_rel_wrap M CH CS RCH W REM cps cmd k cargs vars conts wps wmd
               wargs vars' conts' (Hconts k Hk) Hvars HW Hwrap). }
    split.
    { cbn [with_continuation ap_exn_continuation].
      rewrite (resolve_other (ap_exn_continuation ap) wc),
        (resolve_other (ap_exn_continuation ap) w)
        by (intros He; first [ exact (Hwc (eq_sym He))
                             | exact (Hw (eq_sym He)) ]).
      exact (Hconts _ Hke). }
    split; [ exact Hst | repeat split ].
  - (* A call that may over-apply. *)
    pose proof (over_call_step D0 ap f vars vars' conts conts' S S' rcd
                  Hcallee Hf Hkind Hmy Hargs Hextra Hk Hke Hvars Hconts Hst
                  Hflow Hrem)
      as Hc.
    destruct (step (State (Apply ap) vars conts S empty_heap rcd))
      as [s1 | k ws hp1 rd1 | ]; [ | | exact I ].
    + destruct Hc as (t1 & Ht1 & Hrel). right. left. exists t1.
      split; [ exact Ht1 | exact Hrel ].
    + exists ws, rd1. split; [ exact Hc | reflexivity ].
Qed.

(* Both runs at a call that returns through a wrapper of each: both enter
   related code, the frames of the wrappers related. *)
Lemma call_answered (s t : state) :
  related_call M CH CS RCH W REM s t ->
  answered (related M CH CS RCH W REM) eq measure s t.
Proof.
  destruct s as [p vars conts S hp rcd],
    t as [q vars' conts' S' hp' rcd'].
  intros (ap & k2 & id & ci & Hp & Hq & Hok & Hvars & Hk & Hke & Hst & Hhp
          & Hhp' & Hrcd).
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst p q hp hp' rcd'. unfold answered.
  pose proof (call_entered ap k2 id ci vars vars' conts conts' S S' rcd Hok
                Hvars Hst)
    as Hc.
  destruct (step (State (Apply ap) vars conts S empty_heap rcd))
    as [s1 | | ]; [ | contradiction | exact I ].
  destruct Hc as (cc & cc' & hv & hv' & extra & -> & Ht & Hrel & Hcbody).
  right. left. eexists. split; [ exact Ht | ].
  left. exists (code_dead ci cc).
  split; [ exact Hcbody | split; [ exact Hrel | ] ].
  split; [ | split; [ exact Hst | repeat split ] ].
  apply code_conts_rel;
    [ intros _; exact Hk | apply cont_rel_with_extra; exact Hke ].
Qed.

(* Both runs within a wrapper: they apply a continuation related to the
   same values. *)
Lemma pending_answered (s t : state) :
  related_pending M CH CS RCH W REM s t -> PGP.step_rem_ok REM s ->
  answered (related M CH CS RCH W REM) eq measure s t.
Proof.
  destruct s as [p vars conts S hp rcd],
    t as [q vars' conts' S' hp' rcd'].
  intros (ac & ac' & Hp & Hq & Hk & Hc & Hargs & _ & Hst & Hhp & Hhp'
          & Hrcd) Hrem.
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst p q hp hp' rcd'. unfold answered.
  unfold PGP.step_rem_ok in Hrem.
  cbn [state_expr state_vars state_conts state_statics] in Hrem.
  rewrite step_apply_cont.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
    [ | exact I ].
  unfold apply_continuation.
  pose proof (cont_answered _ _ vs S S' rcd Hc Hst (Hrem vs eq_refl)) as Ha.
  destruct (apply_cont_value (resolve (ac_continuation ac) conts) vs S
              empty_heap rcd)
    as [s1 | k ws hp1 rd1 | ]; [ | | exact I ].
  - destruct Ha as (t1 & Ht1 & Hrel). right. left. exists t1.
    split; [ | exact Hrel ].
    rewrite step_apply_cont, (Hargs vs eq_refl). unfold apply_continuation.
    rewrite Hk. exact Ht1.
  - exists ws, rd1. split; [ | reflexivity ].
    rewrite step_apply_cont, (Hargs vs eq_refl). unfold apply_continuation.
    rewrite Hk. exact Ha.
Qed.

Lemma related_answered (s t : state) :
  related M CH CS RCH W REM s t -> flow_ok s -> PGP.step_rem_ok REM s ->
  answered (related M CH CS RCH W REM) eq measure s t.
Proof.
  intros [Hm | [Hc | Hp]] Hflow Hrem;
    [ exact (main_answered s t Hm Hflow Hrem) | exact (call_answered s t Hc)
    | exact (pending_answered s t Hp Hrem) ].
Qed.


(* [flow_ok] of every state that a state comes to, and what is left of an
   over-application calls code of [REM] only (PGP.step_rem_ok). *)
Definition flow_run (s : state) : Prop :=
  ∀ (n : nat) (s' : state),
    run n s = Step s' -> flow_ok s' ∧ PGP.step_rem_ok REM s'.

(* Related terms, run on their own, refine each other, where the flow of
   the run of the first is as [RCH] says. *)
Lemma rr_refines_flow (p q : expr) :
  rr M CH CS RCH W REM [] p q ->
  flow_run (State p [] [] empty_static_env empty_heap no_reads) ->
  refines (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hrr Hfl.
  apply (shrinking_refines
           (fun s0 t0 => related M CH CS RCH W REM s0 t0 ∧ flow_run s0) eq
           measure).
  - intros s0 t0 [Hr Hfl0].
    pose proof (related_answered s0 t0 Hr (proj1 (Hfl0 0 s0 eq_refl))
                  (proj2 (Hfl0 0 s0 eq_refl)))
      as Ha.
    unfold answered in Ha |- *.
    destruct (step s0) as [s1 | k vs hp rd1 | ] eqn:Hs;
      [ | exact Ha | exact I ].
    assert (Hfl1 : flow_run s1).
    { intros n s2 Hn. apply (Hfl0 (S n)). rewrite run_succ, Hs. exact Hn. }
    destruct Ha
      as [H | [(t1 & Ht & Hr1) | [(Hlt & Hr1) | (t1 & n & Ht & Hr1)]]].
    + left. exact H.
    + right. left. exists t1. exact (conj Ht (conj Hr1 Hfl1)).
    + right. right. left. exact (conj Hlt (conj Hr1 Hfl1)).
    + right. right. right. exists t1, n. exact (conj Ht (conj Hr1 Hfl1)).
  - intros vs vs' ->. reflexivity.
  - split; [ | exact Hfl ].
    left. exists []. split; [ exact Hrr | split; [ exact env_rel_c_empty | ] ].
    split; [ intros k _; constructor | ].
    split; [ | repeat split ].
    split; [ reflexivity | split; [ reflexivity | ] ].
    intros id c Hc. discriminate Hc.
Qed.

(* Where the flow may bring any code anywhere, and what is left of an
   over-application may call any code, it is as [RCH] and [REM] say: the
   partial applications are those of the machine. *)
Lemma flow_run_total (p : expr) :
  (∀ (b : bool) (f : variable) (id : code_id), RCH b f id) ->
  (∀ id : code_id, REM id) ->
  flow_run (State p [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hall Hrall n s Hrun.
  split; [ | exact (PGP.step_rem_ok_total REM s Hrall) ].
  intros ap f v _ _ Hv. split.
  - intros cl applied aa ->.
    pose proof (PGP.pa_inv_run n _ s (PGP.pa_inv_initial p) Hrun)
      as (Hvars & _).
    exact (proj1 (proj1 (PGP.pa_ok_partial _ _ _ _)
                    (PGP.env_pa_ok_of _ _ Hvars f _ Hv))).
  - intros id slots. split; intros _; apply Hall.
Qed.

(* The same where the flow may bring any code anywhere. *)
Lemma rr_refines (p q : expr) :
  (∀ (b : bool) (f : variable) (id : code_id), RCH b f id) ->
  (∀ id : code_id, REM id) ->
  rr M CH CS RCH W REM [] p q ->
  refines (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hall Hrall Hrr.
  exact (rr_refines_flow p q Hrr (flow_run_total p Hall Hrall)).
Qed.

End Simulation.

(* ======================================================================== *)
(* What other files read of the relation                                    *)
(* ======================================================================== *)

(* The other files read [related] only through the lemmas of this
   section and [related_answered], never through its shape, which leaves
   the relation free to have more kinds of states (as one within a
   wrapper of both runs). *)

(* Related states bind the same static values. *)
Lemma related_static_values (M : list (node * unboxed_fields))
    (CH : variable -> option closure_repr)
    (CS : code_id -> option code_info)
    (RCH : bool -> variable -> code_id -> Prop) (W : variable -> Prop)
    (REM : code_id -> Prop)
    (s t : state) :
  related M CH CS RCH W REM s t ->
  static_values (state_statics t) = static_values (state_statics s).
Proof.
  intros [(D & _ & _ & _ & [Hv _] & _)
         | [(_ & _ & _ & _ & _ & _ & _ & _ & _ & _ & [Hv _] & _)
           | (_ & _ & _ & _ & _ & _ & _ & _ & [Hv _] & _)]];
    exact Hv.
Qed.

(* The two terms related, run from empty environments, are related. *)
Lemma related_initial (M : list (node * unboxed_fields))
    (CH : variable -> option closure_repr)
    (CS : code_id -> option code_info)
    (RCH : bool -> variable -> code_id -> Prop) (W : variable -> Prop)
    (REM : code_id -> Prop)
    (p q : expr) :
  rr M CH CS RCH W REM [] p q ->
  related M CH CS RCH W REM (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hrr. left. exists []. split; [ exact Hrr | ].
  split; [ exact (env_rel_c_empty M CH) | ].
  split; [ intros k _; constructor | ].
  split; [ | repeat split ].
  split; [ reflexivity | split; [ reflexivity | ] ].
  intros id c Hc. discriminate Hc.
Qed.

(* The environments of related states are related at the names that are
   not of [W]: a plain name has one value in both, and a changed one
   closures related by [crel_exact]. *)
Lemma related_env (M : list (node * unboxed_fields))
    (CH : variable -> option closure_repr)
    (CS : code_id -> option code_info)
    (RCH : bool -> variable -> code_id -> Prop) (W : variable -> Prop)
    (REM : code_id -> Prop)
    (s t : state) :
  related M CH CS RCH W REM s t ->
  env_rel_cw M CH W (state_vars s) (state_vars t).
Proof.
  intros [(D & _ & Henv & _)
         | [(_ & _ & _ & _ & _ & _ & _ & Henv & _)
           | (_ & _ & _ & _ & _ & _ & _ & Henv & _)]];
    [ exact (env_rel_c_cw M CH W _ _ Henv)
    | exact (env_rel_c_cw M CH W _ _ Henv) | exact Henv ].
Qed.

(* At a [Let_static] of the first run, the second binds static bindings
   related to those of the first. *)
Lemma related_let_static (M : list (node * unboxed_fields))
    (CH : variable -> option closure_repr)
    (CS : code_id -> option code_info)
    (RCH : bool -> variable -> code_id -> Prop) (W : variable -> Prop)
    (REM : code_id -> Prop)
    (s t : state) (bs : list static_binding) (body : expr) :
  related M CH CS RCH W REM s t -> state_expr s = Let_static bs body ->
  ∃ bs' : list static_binding, rr_bindings M CH CS RCH W REM bs bs'.
Proof.
  intros [(D & Hrr & _)
         | [(ap & _ & _ & _ & Hs & _) | (ac & _ & Hs & _)]] He;
    [ | rewrite He in Hs; discriminate Hs | rewrite He in Hs; discriminate Hs ].
  rewrite He in Hrr. inversion Hrr; subst. eexists. eassumption.
Qed.
