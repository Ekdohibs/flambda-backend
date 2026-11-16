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

(* Binding, at the start of a handler, the parameters that every
   application of the continuation gives the same thing: a normalisation
   that refines.

   This file is to be reviewed: [bind_aliased_params] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   Simplify finds, for each parameter of a continuation, whether every
   path to it brings one and the same value: a constant, a symbol, or a
   variable that is in scope where the continuation is defined
   (simplify/flow/dominator_graph.ml, [dominator_analysis], on the graph
   that has an edge from each parameter to each argument that an
   application gives it). Such a parameter is removed with its
   arguments, and its name is bound to that value by a [Let] at the
   start of the handler (simplify/flow/control_flow_graph.ml,
   [minimize_extra_args_for_one_continuation]: the
   [removed_aliased_params_and_extra_params] and the [lets_to_introduce];
   simplify/simplify_let_cont_expr.ml puts the [Let]s around the rebuilt
   handler). The front end makes such parameters for every mutable
   variable: a loop takes each one in scope as a parameter and passes on
   to itself those that its body does not assign, and the continuation
   after an [if] takes them all, assigned in a branch or not.

   [bind_aliased_params] does the second half of that: it adds the
   [Let]s and leaves the parameters and the arguments, which
   FlambdaUnusedParamsSpec.drop_unused_params then removes, the [Let]
   hiding the parameter from the handler. It finds what a parameter is
   always given by looking at the applications ([aliases_of]):

   - of a non-recursive continuation: the parameter at position [i] is
     given [s] if every application of the continuation in the body of
     its [Let_cont] has [s] as its argument at [i];
   - of a recursive continuation that is alone in its [Let_cont]: if
     every application in the body of the [Let_cont] has [s] at [i], and
     every application in the handler has [s] or the parameter itself
     there, the parameter not being bound again inside the handler.
     Simplify's analysis is of the whole graph; this is what one step of
     it finds, and the rounds of the validation repeat it.

   What stops it:
   - a variable [s] that is bound again where the continuation is in
     scope, or is one of its parameters: it would not mean, where it is
     passed, what it means at the start of the handler ([stable]). A
     variable bound outside that the binders of a term do not reuse
     passes;
   - a continuation that some [Apply] in scope has as its return or
     exception continuation (FlambdaUnusedParamsSpec.returned_to): the
     call applies it to values that the term does not show;
   - an exception handler, which a raise applies;
   - parameters that are not distinct, or not as many as the kinds of
     the metadata, which give the [Let]s their kinds;
   - recursive continuations defined several together;
   - the bodies of code are not gone into
     (FlambdaNormaliseSpec.rewrite_code takes the pass there).

   The result refines the term ([bind_aliased_params_refines_statement]),
   under all environments. Nothing is asked of the names of the term
   beyond what the pass checks itself. A binding of the same name as the
   continuation, inside its scope, is not looked for: the applications
   of that other continuation count as applications of this one, which
   can only stop the pass. [returned_to] and [binds_var] may say yes
   when the answer is no, never the reverse.

   Renaming: the functions here compare names by equality only and bind
   none that the term has not got, so the pass commutes with a renaming.
   That statement is owed to the nominal work and is not made here.

   The proofs are in FlambdaParamAliasesProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaUnusedParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the applications of a continuation give it                          *)
(* ======================================================================== *)

Definition simple_eqb (s s' : simple) : bool :=
  if simple_eq_dec s s' then true else false.

(* Whether an expression may bind the variable: by a [Let], or as a
   parameter of a handler. *)
Fixpoint binds_var (x : variable) (e : expr) : bool :=
  match e with
  | Let y _ _ body => Pos.eqb x y || binds_var x body
  | Let_static _ body => binds_var x body
  | Let_cont_non_recursive _ h body =>
    binds_var_in_handler x h || binds_var x body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => binds_var_in_handler x h) handlers
    || binds_var x body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => false
  end
with binds_var_in_handler (x : variable) (h : continuation_handler) : bool :=
  match h with
  | Handler params _ handler => among x params || binds_var x handler
  end.

(* A [Simple] that means the same wherever the expression goes, as where
   the expression starts: a constant, a symbol, or a variable that the
   expression does not bind. *)
Definition stable (s : simple) (e : expr) : bool :=
  match s with
  | Var x => negb (binds_var x e)
  | Symbol _ | Const _ => true
  end.

(* The arguments of some application of [k] in the expression, if it has
   one. *)
Definition args_if_applies (k : continuation) (ac : apply_cont)
    : option (list simple) :=
  if Pos.eqb (ac_continuation ac) k then Some (ac_args ac) else None.

Definition first_some {A : Type} (a b : option A) : option A :=
  match a with
  | Some _ => a
  | None => b
  end.

Fixpoint some_args (k : continuation) (e : expr) : option (list simple) :=
  match e with
  | Let _ _ _ body | Let_static _ body => some_args k body
  | Let_cont_non_recursive _ h body =>
    first_some (some_args k body) (some_args_in_handler k h)
  | Let_cont_recursive handlers body =>
    first_some (some_args k body)
      (fold_right
         (fun '(_, h) found => first_some (some_args_in_handler k h) found)
         None handlers)
  | Apply _ | Invalid => None
  | Apply_cont ac => args_if_applies k ac
  | Switch _ arms =>
    fold_right
      (fun arm found => first_some (args_if_applies k (snd arm)) found)
      None arms
  end
with some_args_in_handler (k : continuation) (h : continuation_handler)
    : option (list simple) :=
  match h with
  | Handler _ _ handler => some_args k handler
  end.

(* Whether an application, if it is of [k], has at position [i] an
   argument that [ok] accepts. *)
Definition given (k : continuation) (i : nat) (ok : simple -> bool)
    (ac : apply_cont) : bool :=
  negb (Pos.eqb (ac_continuation ac) k)
  || match nth_error (ac_args ac) i with
     | Some arg => ok arg
     | None => false
     end.

(* Whether every application of [k] in the expression has. *)
Fixpoint always_given (k : continuation) (i : nat) (ok : simple -> bool)
    (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => always_given k i ok body
  | Let_cont_non_recursive _ h body =>
    always_given_in_handler k i ok h && always_given k i ok body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => always_given_in_handler k i ok h) handlers
    && always_given k i ok body
  | Apply _ | Invalid => true
  | Apply_cont ac => given k i ok ac
  | Switch _ arms => forallb (fun arm => given k i ok (snd arm)) arms
  end
with always_given_in_handler (k : continuation) (i : nat)
    (ok : simple -> bool) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => always_given k i ok handler
  end.

(* For each parameter from position [i] on, what [alias_at] answers of
   it. *)
Fixpoint aliases_from (i : nat) (params : list variable)
    (alias_at : nat -> variable -> option simple) : list (option simple) :=
  match params with
  | [] => []
  | param :: params =>
    alias_at i param :: aliases_from (S i) params alias_at
  end.

(* What each parameter of the continuation [k] is given by every
   application, [None] where that is not one thing. [whole] is the
   [Let_cont] of [k], [body] its body, and [loop] the expression of the
   handler if [k] is recursive: there the parameter itself may be passed
   on. *)
Definition aliases_of (k : continuation) (params : list variable)
    (metadata : handler_metadata) (loop : option expr) (body whole : expr)
    : list (option simple) :=
  if
    returned_to k whole || hm_is_exn_handler metadata
    || negb (distinct params)
    || negb (Nat.eqb (length (hm_params_kinds metadata)) (length params))
  then []
  else
    match some_args k body with
    | None => []
    | Some args =>
      aliases_from 0 params
        (fun i param =>
           match nth_error args i with
           | None => None
           | Some s =>
             if
               stable s whole
               && always_given k i (fun arg => simple_eqb arg s) body
               && match loop with
                  | None => true
                  | Some handler =>
                    always_given k i
                      (fun arg =>
                         simple_eqb arg s || simple_eqb arg (Var param))
                      handler
                    && negb (binds_var param handler)
                  end
             then Some s
             else None
           end)
    end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The parameters that are always given one thing bound to it, around
   the handler. *)
Fixpoint bind_aliases (params : list variable) (kinds : list flambda_kind)
    (aliases : list (option simple)) (handler : expr) : expr :=
  match params, kinds, aliases with
  | param :: params, kind :: kinds, alias :: aliases =>
    let handler := bind_aliases params kinds aliases handler in
    match alias with
    | Some s => Let param kind (Simple s) handler
    | None => handler
    end
  | _, _, _ => handler
  end.

Definition alone {A : Type} (l : list A) : bool :=
  match l with
  | [_] => true
  | _ => false
  end.

Fixpoint bind_aliased_params (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (bind_aliased_params body)
  | Let_static bindings body => Let_static bindings (bind_aliased_params body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata
         (bind_aliases params (hm_params_kinds metadata)
            (aliases_of k params metadata None body e)
            (bind_aliased_params handler)))
      (bind_aliased_params body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, bind_aliased_params_in_loop (alone handlers) k body e h))
         handlers)
      (bind_aliased_params body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with bind_aliased_params_in_loop (alone : bool) (k : continuation)
    (body whole : expr) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let aliases :=
      if alone
      then aliases_of k params metadata (Some handler) body whole
      else []
    in
    Handler params metadata
      (bind_aliases params (hm_params_kinds metadata) aliases
         (bind_aliased_params handler))
  end.

Definition bind_aliased_params_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (bind_aliased_params e).
